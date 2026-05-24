import os
import io
import sys
import joblib
import smtplib
import requests
import secrets
import pandas as pd
import json
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from celery import Celery
from twilio.rest import Client
from groq import Groq
from dotenv import load_dotenv
from flask_apscheduler import APScheduler
from sqlalchemy.exc import IntegrityError

from thefuzz import process 
from scipy.spatial import distance
# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# We import the custom class from its config file as per your main setup
from models_config import HybridYieldModel

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'super-secret-maize-key'),
    SQLALCHEMY_DATABASE_URI='sqlite:///farmers.db',
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0',
    SCHEDULER_API_ENABLED=True
)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
scheduler = APScheduler()

# --- LOADING THE HYBRID MODEL ---
try:
    # Ensuring the custom class is recognized by the unpickler
    sys.modules['__main__'].HybridYieldModel = HybridYieldModel
    model = joblib.load('val_hybrid_crop_model.pkl')
    print("Hybrid Model loaded successfully!")
except Exception as e:
    print(f"Error loading hybrid model: {e}")
    model = None

# --- EXTERNAL SERVICES SETUP ---
twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")) if os.getenv("TWILIO_ACCOUNT_SID") else None
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=60.0)
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# --- DATABASE MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), default='farmer')

class Farmer(db.Model):
    phone = db.Column(db.String(20), primary_key=True)
    user_email = db.Column(db.String(100))
    name = db.Column(db.String(100))
    state = db.Column(db.String(50))
    district = db.Column(db.String(50))
    plots = db.Column(db.Float)
    hectare = db.Column(db.Float)
    lon = db.Column(db.Float)
    lat = db.Column(db.Float)
    ph = db.Column(db.Float)
    clay = db.Column(db.Float)
    sand = db.Column(db.Float)
    silt = db.Column(db.Float)
    planting_date = db.Column(db.String(20))
    language = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, default=datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class SavedChat(db.Model):
    """New: For Farmer Persistent AI Sessions from secondary app.py"""
    id = db.Column(db.Integer, primary_key=True)
    phone = db.Column(db.String(20))
    role = db.Column(db.String(20))
    content = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.now)

class OutreachLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    farmer_phone = db.Column(db.String(20))
    farmer_name = db.Column(db.String(100))
    category = db.Column(db.String(50))
    message = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    temp_avg = db.Column(db.Float)
    precipitation = db.Column(db.Float)
    predicted_yield = db.Column(db.Float)

class ChatHistory(db.Model):
    """Admin-wide chat history"""
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)

class PasswordResetToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    token = db.Column(db.String(100), unique=True)
    expiry = db.Column(db.DateTime)

# --- CELERY WORKER ---
def make_celery(app):
    celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'], broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context(): return self.run(*args, **kwargs)
    celery.Task = ContextTask
    return celery

celery = make_celery(app)

@celery.task(name='app.background_notify')
def background_notify(phone, email, message):
    try:
        # Twilio SMS
        phone = str(phone).strip()
        if phone:
            sms_body = message if len(message) <= 159 else message[:157] + "..."
            # if twilio_client:
            #     twilio_client.messages.create(body=sms_body, from_=TWILIO_FROM, to=phone)

        # Email Notification
        email = str(email).strip()
        if email:
            sender_email = "igbokwedanielc@gmail.com"
            full_msg = (f"From: {sender_email}\nTo: {email}\nSubject: Maize-TechAI\n"
                        f"MIME-Version: 1.0\nContent-Type: text/plain; charset=utf-8\n\n{message}")
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, EMAIL_PASSWORD)
                server.sendmail(sender_email, email, full_msg.encode('utf-8'))
                print(f"Notification sent to {email} successfully!")
        return "Success"
    except Exception as e: return f"Error: {str(e)}"

# --- WEATHER SERVICE (ENHANCED CACHING) ---
def get_weather(lon, lat):
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"],
            "models": "best_match", "timezone": "auto", "past_days": 1, "forecast_days": 7,
        }
        responses = openmeteo.weather_api(url, params=params, timeout=10)
        response = responses[0]

        daily = response.Daily()

        daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
        daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
        daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
        daily_wind_speed_10m_max = daily.Variables(3).ValuesAsNumpy()

        daily_data = {"date": pd.date_range(
            start = pd.to_datetime(daily.Time() + response.UtcOffsetSeconds(), unit = "s", utc = True),
            end =  pd.to_datetime(daily.TimeEnd() + response.UtcOffsetSeconds(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()),
            inclusive = "left"
        )}

        daily_data["temperature_2m_max"] = daily_temperature_2m_max
        daily_data["temperature_2m_min"] = daily_temperature_2m_min
        daily_data["precipitation_sum"] = daily_precipitation_sum
        daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max

        daily_mean = (daily_temperature_2m_max + daily_temperature_2m_min) / 2
        calc_avg = lambda x: round(float(sum(x) / len(x)), 2)
        
        result = {
            "longitude": lon, "latitude": lat, "Average_avg-Temp": calc_avg(daily_mean),
            "Average-Min Temp": calc_avg(daily_temperature_2m_min), "Average-max-temp": calc_avg(daily_temperature_2m_max),
            "avg-precipitation": round(calc_avg(daily_precipitation_sum) * 30, 2), # Scaled 
            "avg-windSpeed": round(calc_avg(daily_wind_speed_10m_max) / 10, 2) # Scaled
        }
        print(f"Avg Precipitation: {result['avg-precipitation']} mm")
        print(f"Avg Wind Speed: {result['avg-windSpeed']} m/s")
        print(f"Weather data retrieved: {result}")
        return result
    except Exception as e:
        print(f"Error retrieving weather data, using defaults: {e}")
        return {"Average_avg-Temp": 27.0, "Average-Min Temp": 22.0, "Average-max-temp": 32.0, "avg-precipitation": 155.20, "avg-windSpeed": 1.66}

# --- YIELD PREDICTION ---
def predict_yield(f, w, log_type="Analysis"):
    if model is None: return {"Estimated_Yield_MT": 0.0, "Soil_Health_Status": "N/A", "Computed_Hectares": 0.0}
    
    # Handle both object (SQLAlchemy) and dict (Simulation) inputs
    is_dict = isinstance(f, dict)
    
    input_df = pd.DataFrame([{
        'longitude': f['lon'] if is_dict else f.lon, 
        'latitude': f['lat'] if is_dict else f.lat,
        'Average_avg-Temp': w['Average_avg-Temp'], 
        'Average-Min Temp': w['Average-Min Temp'],
        'Average-max-temp': w['Average-max-temp'], 
        'avg-precipitation': w['avg-precipitation'], 
        'avg-windSpeed': w['avg-windSpeed'],
        'PH': f['ph'] if is_dict else f.ph, 
        'Clay': f['clay'] if is_dict else f.clay, 
        'Sand': f['sand'] if is_dict else f.sand, 
        'Silt': f['silt'] if is_dict else f.silt,
        'State': f['state'] if is_dict else f.state, 
        'District': f['district'] if is_dict else f.district, 
        'Hectare': f['hectare'] if is_dict else f.hectare
    }])
    
    res = model.predict(input_df)
    return res

# --- ADVISORY ENGINE ---
def get_extensive_advisory(f, w, res):
    p_date = datetime.strptime(f.planting_date, '%Y-%m-%d')
    days_passed = (datetime.now() - p_date).days
    lang = str(f.language).strip().capitalize()
    soil_status = res['Soil_Health_Status']
    advice = []
    
    # --- TRANSLATION DICTIONARY ---
    translations = {
        "English": {
            "harvest_days": "🗓️ {} days remaining until estimated harvest.",
            "harvest_now": "🗓️ Harvest Today.",
            "harvest_passed": "🗓️ {} days past the estimated harvest date.",
            "risk": "📉 Yield Risk: Prediction is {} MT lower than potential.",
            "good": "📈 Great Outlook: Current weather is boosting yield potential!"
                       " - Prediction is {} MT higher than potential.",
            "stable": "✅ Stable: Yield is on track with initial predictions.",
            "acidic": "⚠️ Soil acidic! Apply lime or wood ash.",
            "alkaline": "⚠️ Soil alkaline! Use Ammonium-based fertilizers.",
            "optimal_ph": "✅ pH is optimal for Maize growth.",
            "sown": "🌱 STATUS: Sown. Keep soil moist.",
            "emergence": "🌱 STATUS: Emergence. Check sprouting.",
            "water_seed": "💧 Water needed for young seedlings.",
            "veg": "🌿 STATUS: Vegetative. Time for weeding and NPK.",
            "pest_faw": "🪲 PEST ALERT: Fall Armyworm risk!",
            "tasseling": "🌽 STATUS: Tasseling & Silking. Critical water stage.",
            "water_crit": "💧 CRITICAL: Increase irrigation now.",
            "pest_stem": "🪲 PEST ALERT: Monitor for Stem Borers.",
            "maturity": "🌾 STATUS: Maturity. Prepare for harvest.",
            "rain_alert": "🚫 RAIN ALERT: Heavy rain. Do not fertilize today."
        },
        "Igbo": {
            "harvest_days": "🗓️ Ụbọchị {} fọdụrụ tupu owuwe ihe ubi.",
            "harvest_now": "🗓️ Weghachi ihe ubi taa.",
            "harvest_passed": "🗓️ Ụbọchị {} agafela kemgbe ụbọchị atụmanya iwe ihe ubi.",
            "risk": "📉 Ihe ize ndụ: Ihe ubi ga-agbadata site na {} MT.",
            "good": "📈 Akụkọ ọma: Ihu igwe na-enyere ihe ubi aka nke ọma!",
            "stable": "✅ Ọ dị mma: Ihe ubi gị dị n'ụzọ.",
            "acidic": "⚠️ Ala gbara ụka! Tinye nzu (lime) ma ọ bụ ntụ nkụ.",
            "alkaline": "⚠️ Ala nwere nnu! Jiri fatịlaịza Ammonium.",
            "optimal_ph": "✅ pH ala gị dị mma maka ọka.",
            "sown": "🌱 Ọkwa: Akụrụ akụ. Debe ala ka ọ dị mmiri mmiri.",
            "emergence": "🌱 Ọkwa: Ọka amipụtala. Lelee ma ha pupụtara nke ọma.",
            "water_seed": "💧 Mmiri dị mkpa maka obere ọka ndị a.",
            "veg": "🌿 Ọkwa: Oge ịkpachasị ahịhịa na itinye fatịlaịza NPK.",
            "pest_faw": "🪲 NDỌKWA: Akpụkpọ anụ (Fall Armyworm) nwere ike ịwakpo!",
            "tasseling": "🌽 Ọkwa: Oge ntoputa. Mmiri dị ezigbo mkpa ugbu a.",
            "water_crit": "💧 MKPA: Tinyekwu mmiri ka ihe ubi ghara imebi.",
            "pest_stem": "🪲 NDỌKWA: Lelee maka ụmụ ahụhụ (Stem Borers).",
            "maturity": "🌾 Ọkwa: Oge owuwe ihe ubi. Kwadebe maka ịghọrọ ihe ubi.",
            "rain_alert": "🚫 NDỌKWA: Oke mmiri ozuzo. Atinyela fatịlaịza taa."
        },
        "Hausa": {
            "harvest_days": "🗓️ Sauran kwanaki {} kafin girbi.",
            "harvest_now": "🗓️ Yi girbi a yau.",
            "harvest_passed": "🗓️ Kwana {} sun wuce tun ranar girbin da aka kiyasta.",
            "risk": "📉 Hadarin amfanin gona: Zai ragu da {} MT.",
            "good": "📈 Kyakkyawan fata: Yanayi yana taimakon masara sosai!",
            "stable": "✅ Daidai: Amfanin gona yana tafiya yadda ya kamata.",
            "acidic": "⚠️ Kasar tana da tsami! Yi amfani da gawayi ko alli (lime).",
            "alkaline": "⚠️ Kasar tana da gishiri! Yi amfani da takin Ammonium.",
            "optimal_ph": "✅ pH din kasar yana da kyau ga masara.",
            "sown": "🌱 Matsayi: An shuka. Kasance da danshi a kasa.",
            "emergence": "🌱 Matsayi: Masara ta fito. Duba girman su.",
            "water_seed": "💧 Ana bukatar ruwa ga kananan shuka.",
            "veg": "🌿 Matsayi: Lokacin yin ciyawa da saka takin NPK.",
            "pest_faw": "🪲 GARGADI: Akwai hadarin kwarin Fall Armyworm!",
            "tasseling": "🌽 Matsayi: Lokacin fitar gashin masara. Ruwa yana da muhimmanci.",
            "water_crit": "💧 MUHIMMI: Kara yawan ruwa don gudun asarar amfani.",
            "pest_stem": "🪲 GARGADI: Duba kwarin Stem Borers.",
            "maturity": "🌾 Matsayi: Masara ta nuna. Shirya don girbi.",
            "rain_alert": "🚫 GARGADI: Ruwan sama mai karfi. Kada a saka taki yau."
        },
        "Yoruba": {
            "harvest_days": "🗓️ O ku ọjọ {} ki ẹ to kórè àgbàdo yín.",
            "harvest_now": "🗓️ Kórè lónìí.",
            "harvest_passed": "🗓️ Ọjọ́ {} ti kọjá látìgbà tí a fojúbà láti kórè.",
            "risk": "📉 Ewu fún ìkórè: Ó lè dínkù ní {} MT nítorí ojú ọjọ́.",
            "good": "📈 Ìròyìn ayọ̀: Ojú ọjọ́ dára fún ìdàgbàsókè àgbàdo yín!",
            "stable": "✅ Ó wà ní ìbámu: Ìkórè yín ń lọ dáradára.",
            "acidic": "⚠️ Ilẹ̀ yín ní ekikan! Ẹ lo 'lime' tabi eérú igi.",
            "alkaline": "⚠️ Ilẹ̀ yín ní iyọ̀ jù! Ẹ lo ajílẹ̀ Ammonium.",
            "optimal_ph": "✅ pH ilẹ̀ yín dára fún àgbàdo.",
            "sown": "🌱 Ìpele: Ẹ ti fúnrúgbìn. Ẹ rí i pé ilẹ̀ rin dáradára.",
            "emergence": "🌱 Ìpele: Àgbàdo ti ń yọ. Ẹ rí i pé gbogbo rẹ̀ yọ dáadáa.",
            "water_seed": "💧 Omi ṣe pàtàkì fún àgbàdo tó ṣẹ̀ṣẹ̀ ń yọ.",
            "veg": "🌿 Ìpele: Ìgbà láti roko àti láti lo ajílẹ̀ NPK.",
            "pest_faw": "🪲 ÌKILỌ̀: Ewu kòkòrò 'Fall Armyworm' wà!",
            "tasseling": "🌽 Ìpele: Ìgbà tó ń yọ òdòdó. Omi ṣe pàtàkì gan-an.",
            "water_crit": "💧 PÀTÀKÌ: Ẹ bù omi sí i kí ìkórè má baà bàjẹ́.",
            "pest_stem": "🪲 ÌKILỌ̀: Ẹ ṣọ́ kòkòrò 'Stem Borers'.",
            "maturity": "🌾 Ìpele: Àgbàdo ti gbó. Ẹ múra fún ìkórè.",
            "rain_alert": "🚫 ÌKILỌ̀: Òjò yóò rọ̀ gan-an. Ẹ má ṣe lo ajílẹ̀ lónìí."
        }
    }

    # Default to English if language not supported
    t = translations.get(lang, translations["English"])
    current_yield = res['Estimated_Yield_MT']

    # Harvest Countdown 
    days_to_harvest = max(-120, 120 - days_passed)
    if days_to_harvest == 0:
        advice.append(t["harvest_now"].format(days_to_harvest))
    if days_to_harvest < 0:
        advice.append(t["harvest_passed"].format(days_to_harvest * -1))
    else:
        advice.append(t["harvest_days"].format(days_to_harvest))

    # Soil Health
    if "optimal_ph" in t:
        advice.append(t["optimal_ph"].format(soil_status))
    elif "alkaline" in t:
        advice.append(t["alkaline"].format(soil_status))
    elif "acidic" in t:
        advice.append(t["acidic"].format(soil_status))
    else:
        advice.append(f"🧪 Soil Health: {soil_status}")

    # Yield Analysis
    baseline_weather = {
        "Average_avg-Temp": 27.0, "Average-Min Temp": 19.0, 
        "Average-max-temp": 32.0, "avg-precipitation": 200.5, "avg-windSpeed": 2.0
    }
    baseline_yield = predict_yield(f, baseline_weather)
    baseline_yield = baseline_yield['Estimated_Yield_MT'] if baseline_yield else 0.0
    yield_diff = current_yield - baseline_yield
    print(f"Yield Difference is: {round(yield_diff, 2)} MT/Ha \n(Current: {current_yield} MT/Ha, Baseline: {baseline_yield} MT/Ha)")
    # yield_diff = 0.4
    
    if yield_diff < -0.5:
        advice.append(t["risk"].format(abs(round(yield_diff, 2))))
    elif yield_diff > 0.5:
        advice.append(t["good"].format(abs(round(yield_diff, 2))))
    else:
        advice.append(t["stable"])

    # pH & Soil
    ph = float(f.ph)
    if ph < 5.5: advice.append(t["acidic"])
    elif ph > 7.0: advice.append(t["alkaline"])
    else: advice.append(t["optimal_ph"])

    # Stage-Specific & Pests
    if days_passed < 4:
        advice.append(t["sown"])
    elif 4 <= days_passed <= 14:
        advice.append(t["emergence"])
        if w['avg-precipitation'] < 300.0: advice.append(t["water_seed"])
    elif 15 <= days_passed <= 42:
        advice.append(t["veg"])
        if 26 <= w['Average_avg-Temp'] <= 32: advice.append(t["pest_faw"])
    elif 43 <= days_passed <= 75:
        advice.append(t["tasseling"])
        if w['avg-precipitation'] < 500.0: advice.append(t["water_crit"])
        if w['Average_avg-Temp'] > 30: advice.append(t["pest_stem"])
    elif days_passed > 75:
        advice.append(t["maturity"])

    if w['avg-precipitation'] > 800.0:
        advice.append(t["rain_alert"])

    # Logging
    new_log = OutreachLog(farmer_phone=f.phone, farmer_name=f.name, category = "system",
                          message=f"{f.name} has an estimated yield of {res['Estimated_Yield_MT']} MT/Ha with {res['Soil_Health_Status']} soil, registered on {f.timestamp}. " \
                                        f"Weather conditions show temperatures between {w['Average-Min Temp']}°C and {w['Average-max-temp']}°C, " \
                                        f"with {w['avg-precipitation']}mm of precipitation and wind speeds averaging {w['avg-windSpeed']} m/s. With these Analysis/Recommendation: {advice}", 
                          temp_avg=w['Average_avg-Temp'], precipitation=w['avg-precipitation'], 
                          predicted_yield=res['Estimated_Yield_MT'])
    db.session.add(new_log)
    db.session.commit()

    return advice

# --- ROUTES ---
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid Credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user(); return redirect(url_for('login'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        f_record = Farmer.query.filter_by(user_email=email).first()
        if f_record:
            token_str = secrets.token_urlsafe(32)
            new_token = PasswordResetToken(user_id=f_record.user_id, token=token_str, expiry=datetime.now() + timedelta(minutes=30))
            db.session.add(new_token)
            db.session.commit()
            reset_url = url_for('reset_password', token=token_str, _external=True)
            msg = f"Hello {f_record.name}, click here to reset your password: {reset_url}"
            background_notify.delay(f_record.phone, f_record.user_email, msg)
        flash("If your email is registered, a reset link has been sent.", "info")
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    t_obj = PasswordResetToken.query.filter_by(token=token).first()
    if not t_obj or t_obj.expiry < datetime.now():
        flash("Reset link is invalid or has expired.", "danger")
        return redirect(url_for('login'))
    if request.method == 'POST':
        u_obj = db.session.get(User, t_obj.user_id)
        u_obj.password = generate_password_hash(request.form.get('password'))
        db.session.delete(t_obj)
        db.session.commit()
        flash("Password updated! Please login.", "success")
        return redirect(url_for('login'))
    return render_template('reset_with_token.html', token=token)

# --- MAIN INDEX ROUTE (ADMIN & FARMER DASHBOARDS) ---
@app.route('/')
@login_required
def index():
    if current_user.role == 'admin':
        farmers = Farmer.query.order_by(Farmer.timestamp.desc()).all()
        
        job = scheduler.get_job('weekly_alert')
        if job:
            is_paused = (job.next_run_time is None)
        else:
            is_paused = True
        return render_template('index.html', farmers=farmers, is_paused=is_paused)
    
    else:
        f = Farmer.query.filter_by(user_id=current_user.id).first()
        if not f:
            flash("Farmer profile not found. Please contact Admin.", "danger")
            return redirect(url_for('logout'))
            
        w = get_weather(f.lon, f.lat)
        res = predict_yield(f, w, log_type="Farmer Dashboard View")
        adv = get_extensive_advisory(f, w, res)
        
        return render_template('farmer_dashboard.html', farmer=f, hybrid_res=res, advisory=adv)

# --- ADMIN ANALYTICS & SIMULATOR ROUTE ---
@app.route('/admin-dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        return redirect(url_for('index'))
    
    farmers = Farmer.query.all()
    stats = []
    total_yield = 0
    
    # Baseline weather for deviation calculation
    baseline_weather = {
        "Average_avg-Temp": 27.0, "Average-Min Temp": 19.0, 
        "Average-max-temp": 32.0, "avg-precipitation": 200.5, "avg-windSpeed": 2.0
    }
    
    for f in farmers:
        # Get live weather for each farmer
        w = get_weather(f.lon, f.lat)
        
        # Current Prediction
        res = predict_yield(f, w, log_type="Dashboard Analytics")
        curr_y = res['Estimated_Yield_MT']
        
        # Baseline Prediction (to see if they are doing better or worse than average)
        base_res = predict_yield(f, baseline_weather, log_type="Dashboard Baseline")
        base_y = base_res['Estimated_Yield_MT']
        
        total_yield += curr_y
        
        stats.append({
            'name': f.name,
            'state': f.state,
            'district': f.district,
            'lat': f.lat,
            'lon': f.lon,
            'hectare': f.hectare,
            'ph': f.ph,
            'clay': f.clay,
            'sand': f.sand,
            'yield': curr_y,
            'deviation': round(curr_y - base_y, 2)
        })
    
    # Simple regional sum for the stats cards
    regional_data = {}
    for s in stats:
        regional_data[s['state']] = regional_data.get(s['state'], 0) + s['yield']

    return render_template('admin_dashboard.html', 
                           stats=stats, 
                           total_yield=round(total_yield, 2), 
                           regional=regional_data)


# --- THE SIMULATION BRIDGE  ---
@app.route('/simulate-yield', methods=['POST'])
@login_required
def simulate_yield():
    if current_user.role != 'admin': return jsonify({"error": "Unauthorized"}), 403
    d = request.json
    sim_data = {
        'lon': float(d['lon']), 'lat': float(d['lat']), 'ph': float(d['ph']), 
        'clay': float(d['clay']), 'sand': float(d['sand']), 'silt': float(d['silt']),
        'state': d['state'], 'district': d['district'], 'hectare': float(d['hectare'])
    }
    w = get_weather(sim_data['lon'], sim_data['lat'])
    res = predict_yield(sim_data, w)
    return jsonify({"simulated_yield": res['Estimated_Yield_MT'], "weather": w, "soil_status": res['Soil_Health_Status']})

# --- FARMER PERSISTENT AI  ---
@app.route('/farmer-chat', methods=['POST'])
@login_required
def farmer_chat():
    f = Farmer.query.filter_by(user_id=current_user.id).first()
    user_msg = request.json.get('message', '')
    
    # Fetch saved history from DB
    saved = SavedChat.query.filter_by(phone=f.phone).order_by(SavedChat.timestamp.asc()).all()
    history = [{"role": s.role, "content": s.content} for s in saved]
    
    sys_msg = f"You are an agricultural expert for only{f.name} in {f.district}. Land: {f.hectare}Ha." \
              f"Stick to their data only and do not give out any information about any other farmer or the Admin database." \
              f"Strictly the database of {f.name}."
    messages = [{"role": "system", "content": sys_msg}]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": user_msg})

    completion = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
    ai_res = completion.choices[0].message.content
    
    # Auto-save Farmer chats to the SavedChat table (Persistent Session)
    db.session.add(SavedChat(phone=f.phone, role="user", content=user_msg))
    db.session.add(SavedChat(phone=f.phone, role="assistant", content=ai_res))
    db.session.commit()
    
    return jsonify({"response": ai_res})
    

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if current_user.role != 'admin': 
        return redirect(url_for('index'))
    
    # --- NEW: LOAD LOCATION DATA FOR DROPDOWNS ---
    try:
        df = pd.read_csv('Train_Cropyield_Model.csv')
        mapping = df.groupby('State')['District'].apply(lambda x: sorted(list(set(x)))).to_dict()
        states_list = sorted(mapping.keys())

        # Create a map of { 'DistrictName': [latitude, longitude] }
        # We take the mean (average) coordinates for each district
        coords_df = df.groupby('District')[['latitude', 'longitude']].mean().to_dict('index')
        # Reformat it into a simpler dictionary for JavaScript: { 'AMAC': [9.06, 7.39], ... }
        dist_coords_map = {d: [v['latitude'], v['longitude']] for d, v in coords_df.items()}

    except Exception as e:
        print(f"Error loading location data: {e}")
        states_list = []
        mapping = {}
        dist_coords_map = {}

    if request.method == 'POST':
        d = request.form
        try:
            p_hash = generate_password_hash(d['phone'])
            new_u = User(username=d['phone'], password=p_hash, role='farmer')
            db.session.add(new_u)
            db.session.flush()

            new_f = Farmer(
                phone=d['phone'], user_email=d['user_email'], name=d['name'], 
                state=d['state'], district=d['district'], 
                hectare=float(d['hectare']), plots=float(d['plots']), 
                lon=float(d['lon']), lat=float(d['lat']), 
                ph=float(d['ph']), clay=float(d['clay']), sand=float(d['sand']), silt=float(d['silt']), 
                planting_date=d['planting_date'], language=d['language'], user_id=new_u.id
            )
            db.session.add(new_f) 
            db.session.commit()
            flash(f"Farmer {d['name']} registered!", "success") 
            return redirect(url_for('index'))
            
        except IntegrityError:
            db.session.rollback()
            flash("User already exists !!! (Phone No. already in use)", "danger")
            return render_template('register.html', states=states_list, state_district_map=mapping, district_coords_map=dist_coords_map)

    return render_template('register.html', states=states_list, state_district_map=mapping, district_coords_map=dist_coords_map)


@app.route('/api/get-geo-soil-data')
def get_geo_soil_data():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    # Required for GPS fallback distance calculation
    f_lat, f_lon = float(lat), float(lon)

    # --- PART A: GEOCODING URL ---
    headers = {'User-Agent': 'MaizeTech_Nigeria_App_v1'}
    geo_url = (
        f"https://nominatim.openstreetmap.org/reverse"
        f"?lat={lat}&lon={lon}&format=json"
    )

    try:
        df = pd.read_csv('Train_Cropyield_Model.csv')
        state_district_map = df.groupby('State')['District'].apply(lambda x: sorted(list(set(x)))).to_dict()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        state_district_map = {}
    
    # Initialize variables for the new logic
    matched_state = "Unknown"
    matched_district = "Unknown"
    needs_verification = False
    fuzzy_candidate = {}
    gps_candidate = {}

    try:
        geo_res = requests.get(geo_url, headers=headers, timeout=10).json()
        address = geo_res.get('address', {})
        raw_state = address.get('state', 'Unknown')
        raw_district = (
            address.get('village') or 
            address.get('hamlet') or
            address.get('neighbourhood') or 
            address.get('quarter') or 
            address.get('suburb') or 
            address.get('city') or 
            address.get('town') or 
            address.get('county') or 
            'Unknown'
        )

        # --- MAINTAINED STATE MATCHING ---
        for s in state_district_map.keys():
            if s == "Abuja" and ("Abuja" in raw_state or "Federal Capital Territory" in raw_state):  
                matched_state = s
            elif s.lower() in raw_state.lower():
                matched_state = s
        
        # --- HYBRID DISTRICT MATCHING ---
        if matched_state != "Unknown":
            csv_districts = state_district_map[matched_state]
            state_df = df[df['State'] == matched_state].copy()

            if not state_df.empty:
                # 1. FUZZY CANDIDATE
                f_name, f_score = process.extractOne(raw_district, csv_districts)
                
                # 2. GPS CANDIDATE (Find absolute closest in state)
                state_df['dist'] = state_df.apply(
                    lambda r: distance.euclidean((f_lat, f_lon), (r['latitude'], r['longitude'])), 
                    axis=1
                )
                idx_closest = state_df['dist'].idxmin()
                g_name = state_df.loc[idx_closest, 'District']
                g_dist = state_df.loc[idx_closest, 'dist']
                
                # Calculate distance for the fuzzy choice to compare
                f_dist = state_df[state_df['District'] == f_name]['dist'].min()

                # --- VALIDATION LOGIC ---
                if f_name == g_name:
                    # They agree - proceed normally
                    matched_district = f_name
                    needs_verification = False
                else:
                    # Conflict found (e.g., Name Karu vs Location Gwarinpa)
                    matched_district = f_name  # Default to fuzzy
                    needs_verification = True
                    # Get fuzzy candidate's first available coordinates from CSV
                    f_row = state_df[state_df['District'] == f_name].iloc[0]
                    fuzzy_candidate = {
                        "name": f_name, 
                        "dist": round(f_dist, 4),
                        "lat": round(f_row['latitude'], 4),
                        "lon": round(f_row['longitude'], 4)
                    }
                    gps_row = state_df[state_df['District'] == g_name].iloc[0]
                    gps_candidate = {
                        "name": g_name, 
                        "dist": round(g_dist, 4),
                        "lat": round(gps_row['latitude'], 4),
                        "lon": round(gps_row['longitude'], 4)
                    }

    except Exception as e:
        print(f"Geo Error: {e}")
        matched_state, matched_district = "Unknown", "Unknown"

    # --- PART B: SOIL DATA URL ---
    soil_url = (
        f"https://rest.isric.org/soilgrids/v2.0/properties/query"
        f"?lon={lon}&lat={lat}"
        f"&property=phh2o"
        f"&property=clay"
        f"&property=sand"
        f"&property=silt"
        f"&depth=0-5cm"
        f"&value=mean"
    )
    
    try:
        soil_res = requests.get(soil_url, timeout=10).json()
        layers = soil_res.get('properties', {}).get('layers', [])
        props = {}
        for layer in layers:
            try:
                name = layer.get('name')
                depths = layer.get('depths', [])
                if not depths: continue
                values = depths[0].get('values', {})
                props[name] = values.get('mean')
            except Exception as e:
                print("Layer Parse Error:", e)
    except Exception as e:
        print(f"Soil Error: {e}")
        props = {}

    return jsonify({
        "state": matched_state,
        "district": matched_district,
        "needs_verification": needs_verification,
        "fuzzy_candidate": fuzzy_candidate,
        "gps_candidate": gps_candidate,
        "ph": props.get('phh2o', 0) / 10.0,
        "clay": props.get('clay', 0) / 10.0,
        "sand": props.get('sand', 0) / 10.0,
        "silt": props.get('silt', 0) / 10.0
    })



@app.route('/analyze-farmer/<phone>')
@login_required
def analyze_farmer(phone):
    if current_user.role != 'admin': return redirect(url_for('index'))
    f = db.session.get(Farmer, phone)
    if f:
        w = get_weather(f.lon, f.lat)
        res = predict_yield(f, w, log_type="Admin Manual Analysis")
        recs = get_extensive_advisory(f, w, res)
        flash(f"🔍 ANALYSIS FOR {f.name}: ESTIMATED YIELD 🌱 IS :  {res['Estimated_Yield_MT']} MT/Ha", "primary")
        for rec in recs: flash(rec, "info")
    return redirect(url_for('index')) 

@app.route('/notify-individual/<phone>')
@login_required
def notify_individual(phone):
    if current_user.role != 'admin': return redirect(url_for('index'))
    f = db.session.get(Farmer, phone)
    if f:
        w = get_weather(f.lon, f.lat)
        res = predict_yield(f, w, log_type="Individual Notify Triggered")
        recs = get_extensive_advisory(f, w, res)

        try:
            prompt = (
                      f"Analyze data for {f.name}: Current Yield {res['Estimated_Yield_MT']} MT/Ha, "
                      f"Weather: {w.get('description', 'N/A')}, "
                      f"Current Recommendations: {', '.join(recs)}. "
                      f"Provide a concise 2-sentence summary of what is happening and your top recommendation."
                )
            completion = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a professional agricultural expert."}, 
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7, max_tokens=150 
            )
            ai_summary = completion.choices[0].message.content
        except:
            ai_summary = "Our AI system is currently evaluating your latest data to provide a custom overview."

        # STRUCTURED TEXT MESSAGE
        msg = (
            f"SUMMARY / ANALYSIS FOR {f.name.upper()}\n"
            f"             💡🔎💡🔎💡🔎           \n"
            f"After analyzing the farmland data, this is what we recommend. \n"
            f"AI INSIGHT: {ai_summary}\n\n"
            f"Kindly review the clear points and recommendations below for your current farming cycle:\n"
        )

        for rec in recs:
            low_rec = rec.lower()
            if "days" in low_rec:
                msg += f"DURATION TO HARVEST: {rec}\n"
            elif "soil" in low_rec:
                msg += f"{rec}\n"
            elif "weather" in low_rec or "forecast" in low_rec:
                msg += f"WEATHER OUTLOOK: {rec}\n"
            elif "status" in low_rec or "maturity" in low_rec:
                msg += f"FARMING{rec}\n"
            elif "risk" in low_rec:
                msg += f"CRITICAL POINT: {rec}\n"
            else:
                msg += f"ACTION POINT: {rec}\n"

        msg += f"\nESTIMATED YIELD: {res['Estimated_Yield_MT']} MT/Ha"

        background_notify.delay(f.phone, f.user_email, msg)
        flash(f"✅ Notification sent to {f.name}!", "success")
    return redirect(url_for('index'))

@app.route('/delete/<phone>')
@login_required
def delete(phone):
    if current_user.role != 'admin': return redirect(url_for('index'))
    f = db.session.get(Farmer, phone)
    if f:
        u = db.session.get(User, f.user_id)
        db.session.delete(f)
        if u: db.session.delete(u)
        db.session.commit()
        flash("Farmer profile deleted.", "success")
    return redirect(url_for('index'))


@app.route('/chat', methods=['POST'])
@login_required
def chat():
    if current_user.role != 'admin': return jsonify({"response": "Unauthorized"}), 403
    user_msg = request.json.get('message', '')
    # Fetch recent farmers for context
    farmers = Farmer.query.order_by(Farmer.timestamp.desc()).limit(30).all()
    # Fetch recent logs for context
    logs = OutreachLog.query.order_by(OutreachLog.timestamp.desc()).limit(30).all()
    # Fetch recent chat history for context    
    chat_history = ChatHistory.query.order_by(ChatHistory.timestamp.desc()).limit(100).all()[::-1]
    f_ctx = "FARMERS:\n" + "\n".join(
        [f"- {f.name}, Region: {f.district}, {f.state}." 
         f" Latitude: {f.lat}, Longitude: {f.lon}. Land Size: {f.plots} Plots, {f.hectare} Ha." 
         f" Planted on: {f.planting_date}. Registered on: {f.timestamp}. Soil: PH {f.ph}, Clay {f.clay}%, Sand {f.sand}%, Silt {f.silt}%."
         for f in farmers])
    l_ctx = "LOGS:\n" + "\n".join([f"- {l.message}" for l in logs])
    c_ctx = "CHAT HISTORY:\n" + "\n".join([f"- {h.role.upper()}: {h.content}" for h in chat_history])
    messages = [
        {"role": "system", "content": f"You are MaizeTech AI, a precision agriculture assistant. You are speaking to the System Admin/Manager." 
                                    f"Treat the user as the overseer of the data provided in \n{f_ctx}\n{l_ctx}\n{c_ctx}." 
                                    f"Do not address the user as a farmer, Don't saying any other thing until when asked a question;" 
                                    f"Do not give any analysis until when asked explicitly, Just only introduce yourself shortly and nothing more;" 
                                    f"instead, provide them with the high-level analytics, tracking progress," 
                                    f"alerts they need to manage the farmers and their crops effectively."
                                    f"Use this context: \n{f_ctx}\n{l_ctx}\n{c_ctx}. Be brief and professional."}, 
        {"role": "user", "content": user_msg}]


    try:
        completion = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages)
        ai_res = completion.choices[0].message.content
        db.session.add(ChatHistory(role="user", content=user_msg))
        db.session.add(ChatHistory(role="assistant", content=ai_res))
        db.session.commit()
        return jsonify({"response": ai_res})
    except Exception as e: return jsonify({"response": str(e)})

@app.route('/get-chat-history')
@login_required
def get_chat_history():
    history = ChatHistory.query.order_by(ChatHistory.timestamp.asc()).all()
    return jsonify([{"role": h.role, "content": h.content} for h in history])

@app.route('/clear-chat', methods=['POST'])
@login_required
def clear_chat():
    ChatHistory.query.delete()
    db.session.commit()
    return jsonify({"status": "success"})

@app.route('/clear_log', methods=['POST'])
@login_required
def clear_log():
    OutreachLog.query.delete()
    db.session.commit()
    return jsonify({"status": "success"})

# --- SCHEDULER LOGIC ---
def weekly_sms_email_blast():
    with app.app_context():
        farmers = Farmer.query.all()
        for f in farmers:
            w = get_weather(f.lon, f.lat)
            res = predict_yield(f, w, log_type="Weekly Automated Blast")
            recs = get_extensive_advisory(f, w, res)

            try:
                prompt = (
                        f"Analyze data for {f.name}: Current Yield {res['Estimated_Yield_MT']} MT/Ha, "
                        f"Weather: {w.get('description', 'N/A')}, "
                        f"Current Recommendations: {', '.join(recs)}. "
                        f"Provide a concise 2-sentence summary of what is happening and your top recommendation."
                    )
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a professional agricultural expert."}, 
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7, max_tokens=150 
                )
                ai_summary = completion.choices[0].message.content
            except:
                ai_summary = "Our AI system is currently evaluating your latest data to provide a custom overview."

            # STRUCTURED TEXT MESSAGE
            msg = (
                f"SUMMARY / ANALYSIS FOR {f.name.upper()}\n"
                f"             💡🔎💡🔎💡🔎           \n"
                f"After analyzing the farmland data, this is what we recommend. \n"
                f"AI INSIGHT: {ai_summary}\n\n"
                f"Kindly review the clear points and recommendations below for your current farming cycle:\n"
            )

            for rec in recs:
                low_rec = rec.lower()
                if "days" in low_rec:
                    msg += f"DURATION TO HARVEST: {rec}\n"
                elif "soil" in low_rec:
                    msg += f"{rec}\n"
                elif "weather" in low_rec or "forecast" in low_rec:
                    msg += f"WEATHER OUTLOOK: {rec}\n"
                elif "status" in low_rec or "maturity" in low_rec:
                    msg += f"FARMING{rec}\n"
                elif "risk" in low_rec:
                    msg += f"CRITICAL POINT: {rec}\n"
                else:
                    msg += f"ACTION POINT: {rec}\n"

            msg += f"\nESTIMATED YIELD: {res['Estimated_Yield_MT']} MT/Ha"
            background_notify.delay(f.phone, f.user_email, msg)

@app.route('/trigger-schedule')
@login_required
def trigger_schedule():
    if current_user.role == 'admin':
        weekly_sms_email_blast()
        flash("⚡ Manual Update Cycle Triggered!", "success")
    return redirect(url_for('index'))

@app.route('/start-automated-schedule')
@login_required
def start_automated_schedule():
    if current_user.role == 'admin':
        scheduler.resume_job('weekly_alert')
        flash("✅ Automated 10-minute cycle ACTIVATED!", "success")
    return redirect(url_for('index'))

@app.route('/pause-automated-schedule')
@login_required
def pause_automated_schedule():
    if current_user.role == 'admin':
        scheduler.pause_job('weekly_alert')
        flash("⏸️ Automated weekly cycle PAUSED!", "warning")
    return redirect(url_for('index'))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username=os.getenv('ADMIN_USERNAME')).first():
            db.session.add(User(username=os.getenv('ADMIN_USERNAME'), password=generate_password_hash(os.getenv('ADMIN_PASSWORD')), role='admin'))
            db.session.commit()
    scheduler.init_app(app)
    scheduler.start()
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        if not scheduler.get_job('weekly_alert'):
            scheduler.add_job(id='weekly_alert', func=weekly_sms_email_blast, trigger='interval', minutes=10)
            scheduler.pause_job('weekly_alert')
    app.run(debug=True)
