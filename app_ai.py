# app.py - Complete Streamlit App - FINAL VERSION
"""
STUDENT SCORE PREDICTOR - Streamlit Web App
Loads model, scaler, and dataset from local files
Includes time input with 5/10/15 minute precision options
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🎓 Student Score Predictor",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
        }
        .score-card {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
        }
        .score-number {
            font-size: 5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        .grade-badge {
            font-size: 1.5rem;
            padding: 0.5rem 2rem;
            border-radius: 30px;
            background: rgba(255,255,255,0.2);
            display: inline-block;
        }
        .feature-box {
            background: #f0f2f6;
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.3rem 0;
        }
        .good { border-left: 4px solid #2ecc71; }
        .warning { border-left: 4px solid #f39c12; }
        .bad { border-left: 4px solid #e74c3c; }
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            transition: all 0.3s;
        }
        .stButton>button:hover:not(:disabled) {
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .stButton>button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .time-input-container {
            background: #f8f9fa;
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.3rem 0;
        }
        .increment-selector {
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# GET CURRENT DIRECTORY
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"📂 Script directory: {SCRIPT_DIR}")

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================
@st.cache_resource
def load_model_and_scaler():
    """
    Load the trained model, scaler, and feature columns.
    Uses @st.cache_resource so it only loads once.
    """
    model_path = "/Users/dharsh/Documents/Thesis/SHAP Model/final_exam_model.pkl"
    scaler_path = "/Users/dharsh/Documents/Thesis/SHAP Model/robust_scaler.pkl"
    columns_path = "/Users/dharsh/Documents/Thesis/SHAP Model/model_columns.pkl"
    
    print(f"📂 Looking for model at: {model_path}")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        st.info("Please run 'python train_shap_model.py' first to create the model files.")
        return None, None, None
    
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded successfully")
        feature_columns = joblib.load(columns_path)
        print(f"✅ Feature columns loaded: {len(feature_columns)} columns")
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

# ============================================================
# LOAD DATASET
# ============================================================
@st.cache_data
def load_dataset():
    """Load the student dataset."""
    dataset_path = os.path.join(SCRIPT_DIR, 'student_digital_life.csv')
    print(f"📂 Looking for dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        st.warning(f"⚠️ Dataset not found at: {dataset_path}")
        return None
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        st.warning(f"⚠️ Error loading dataset: {e}")
        return None

# ============================================================
# LOAD EVERYTHING
# ============================================================
model, scaler, feature_columns = load_model_and_scaler()
df = load_dataset()

# Show status in sidebar
if model is not None:
    st.sidebar.success("✅ Model loaded")
else:
    st.sidebar.error("❌ Model not loaded")

if df is not None:
    st.sidebar.success(f"✅ Dataset: {len(df)} students")
else:
    st.sidebar.warning("⚠️ Dataset not loaded")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def format_hours(hours):
    """Format hours as hours and minutes for display"""
    hrs = int(hours)
    mins = int((hours - hrs) * 60)
    if hrs > 0 and mins > 0:
        return f"{hrs}h {mins}m"
    elif hrs > 0:
        return f"{hrs}h"
    elif mins > 0:
        return f"{mins}m"
    else:
        return "0m"

def time_input_combined(label, max_hours, default_hours, key_prefix):
    """
    Create time input with hours and minutes in a single combined input
    Uses a dropdown for minutes with 5, 10, or 15 min increments
    """
    default_hrs = int(default_hours)
    default_mins = int((default_hours - default_hrs) * 60)
    
    # Round default minutes to nearest 5
    default_mins = int(round(default_mins / 5) * 5)
    if default_mins >= 60:
        default_mins = 55
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        hrs = st.number_input(
            f"{label} (Hours)",
            min_value=0,
            max_value=max_hours,
            value=default_hrs,
            step=1,
            key=f"{key_prefix}_hrs"
        )
    
    with col2:
        # Minutes with 5, 10, or 15 min options
        mins = st.selectbox(
            f"{label} (Minutes)",
            options=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            index=default_mins // 5,
            key=f"{key_prefix}_mins",
            help="Select minutes in 5-minute increments"
        )
    
    with col3:
        # Show formatted time
        total_hours = hrs + mins / 60
        st.markdown(f"<div style='margin-top: 25px; font-weight: bold;'>⏱️ {format_hours(total_hours)}</div>", 
                   unsafe_allow_html=True)
    
    return total_hours

def time_input_with_15min(label, max_hours, default_hours, key_prefix):
    """
    Alternative: Time input with 15-minute increments only
    """
    default_hrs = int(default_hours)
    default_mins = int((default_hours - default_hrs) / 0.25) * 15  # Round to nearest 15 min
    if default_mins >= 60:
        default_mins = 45
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        hrs = st.number_input(
            f"{label} (Hours)",
            min_value=0,
            max_value=max_hours,
            value=default_hrs,
            step=1,
            key=f"{key_prefix}_hrs"
        )
    
    with col2:
        mins = st.selectbox(
            f"{label} (Minutes)",
            options=[0, 15, 30, 45],
            index=default_mins // 15,
            key=f"{key_prefix}_mins",
            help="Select minutes in 15-minute increments"
        )
    
    with col3:
        total_hours = hrs + mins / 60
        st.markdown(f"<div style='margin-top: 25px; font-weight: bold;'>⏱️ {format_hours(total_hours)}</div>", 
                   unsafe_allow_html=True)
    
    return total_hours

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_features(input_dict):
    """Apply the same feature engineering as training"""
    df = pd.DataFrame([input_dict])
    
    # Study efficiency
    df["study_efficiency"] = np.clip(
        df["study_hours_per_day"] / (df["sleep_hours"] + 1e-5), 0, 10
    )
    
    # Focus ratio
    df["focus_ratio"] = np.clip(
        df["study_hours_per_day"] / (df["smartphone_usage_hours"] + 1e-5), 0, 20
    )
    
    # Total screen time
    df["total_screen_time"] = (
        df["social_media_hours"] + df["gaming_hours"] + df["streaming_hours"]
    )
    
    # Distraction ratio
    df["distraction_ratio"] = (
        df["total_screen_time"] / (df["study_hours_per_day"] + 1e-5)
    )
    
    # Health score
    df["health_score"] = df["sleep_hours"] + df["exercise_hours"]
    
    # Stress proxy
    df["stress_proxy"] = df["caffeine_intake_cups"] / (df["sleep_hours"] + 1e-5)
    
    # Engagement score
    df["engagement_score"] = (
        df["class_attendance_percent"] + df["assignment_completion_percent"]
    ) / 2
    
    # Balance score
    df["balance_score"] = df["study_hours_per_day"] / (
        df["total_screen_time"] + df["sleep_hours"] + 1e-5
    )
    
    # Motivation effect
    df["motivation_effect"] = df["motivation_level"] * df["study_hours_per_day"]
    
    # Study vs screen
    df["study_vs_screen"] = df["study_hours_per_day"] / (df["total_screen_time"] + 1e-5)
    
    # Sleep efficiency
    df["sleep_efficiency"] = df["sleep_hours"] * df["study_efficiency"]
    
    # Mental pressure
    df["mental_pressure"] = df["stress_proxy"] * df["distraction_ratio"]
    
    # Screen per study hour
    df["screen_per_study_hour"] = df["total_screen_time"] / (df["study_hours_per_day"] + 1e-5)
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Ensure all feature columns are present
    if feature_columns is not None:
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_columns]
    
    return df_encoded

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_score(student_data):
    """Make prediction using the trained model"""
    if model is None or scaler is None:
        return None
    
    try:
        features = engineer_features(student_data)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        return np.clip(prediction, 0, 100)
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None

# ============================================================
# GENERATE EXPLANATION
# ============================================================
def generate_explanation(student_data, score):
    """Generate simple explanation of strengths and weaknesses"""
    reasons = []
    improvements = []
    
    # Study hours
    study = student_data['study_hours_per_day']
    if study >= 6:
        reasons.append(f"Studies {format_hours(study)} daily - excellent dedication!")
    elif study >= 4:
        reasons.append(f"Studies {format_hours(study)} daily - good amount")
    else:
        improvements.append(f"Only {format_hours(study)} study - needs 4-5 hours")
    
    # Sleep
    sleep = student_data['sleep_hours']
    if 7 <= sleep <= 8:
        reasons.append(f"{format_hours(sleep)} sleep - optimal for learning")
    elif sleep < 6:
        improvements.append(f"Only {format_hours(sleep)} sleep - affects focus")
    
    # Screen time (using max of gaming/streaming for overlap)
    max_entertainment = max(student_data['gaming_hours'], student_data['streaming_hours'])
    screen_total = student_data['social_media_hours'] + student_data['smartphone_usage_hours'] + max_entertainment
    
    if screen_total <= 3:
        reasons.append(f"Low screen time ({format_hours(screen_total)}) - great focus")
    elif screen_total > 6:
        improvements.append(f"High screen time ({format_hours(screen_total)}) - major distraction")
    
    # Attendance
    attendance = student_data['class_attendance_percent']
    if attendance >= 90:
        reasons.append(f"Excellent attendance ({attendance}%)")
    elif attendance < 75:
        improvements.append(f"Low attendance ({attendance}%)")
    
    # Assignments
    assignments = student_data['assignment_completion_percent']
    if assignments >= 90:
        reasons.append(f"Completes {assignments}% of assignments")
    elif assignments < 70:
        improvements.append(f"Only {assignments}% assignments completed")
    
    # Motivation
    motivation = student_data['motivation_level']
    if motivation >= 8:
        reasons.append(f"High motivation ({motivation}/10)")
    elif motivation <= 4:
        improvements.append(f"Low motivation ({motivation}/10)")
    
    # Exercise
    exercise = student_data['exercise_hours']
    if exercise >= 1:
        reasons.append(f"Exercises {format_hours(exercise)} daily - good for brain health")
    
    # Caffeine
    caffeine = student_data['caffeine_intake_cups']
    if caffeine > 3:
        improvements.append(f"High caffeine ({caffeine} cups) - may affect sleep")
    
    return reasons, improvements

# ============================================================
# SESSION STATE
# ============================================================
def init_session_state():
    if 'points' not in st.session_state:
        st.session_state.points = 0
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = 0
    if 'badges' not in st.session_state:
        st.session_state.badges = []
    if 'history' not in st.session_state:
        st.session_state.history = []

init_session_state()

# ============================================================
# SIDEBAR
# ============================================================
def show_sidebar():
    with st.sidebar:
        st.markdown("### 🎮 Dashboard")
        st.markdown(f"**📊 Predictions:** {st.session_state.predictions_made}")
        
        st.markdown("---")
        menu = ["🏠 Home", "📊 Predict"]
        choice = st.radio("Navigate", menu, index=0)
        
        if st.button("🔄 Reset Prediction"):
            st.session_state.predictions_made = 0
            st.session_state.history = []
            st.rerun()
        
        return choice

# ============================================================
# HOME PAGE
# ============================================================
def home_page():
    st.markdown('<div class="main-header">🎓 Student Score Predictor</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3>📊 Make Predictions</h3>
            <p>Enter student data and get AI-powered exam score predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3>🔮 Do you trust AI?</h3>
            <p>Check if you can trust the AI generated score!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if df is not None:
        st.markdown(f"### 📊 Dataset Info")
        st.write(f"**Total Students:** {len(df)}")
        st.write(f"**Features:** {len(df.columns)}")
        st.dataframe(df.head(), use_container_width=True)

# ============================================================
# PREDICTION PAGE - WITH COMBINED TIME INPUT
# ============================================================
def prediction_page():
    st.markdown('<div class="main-header">📊 Make a Prediction</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Model not loaded. Please check the model files.")
        return
    
    with st.container():
        st.markdown("### 📋 Student Information")
        st.info("ℹ️ **Note:** Total hours cannot exceed 24 hours per day. Gaming and Streaming can overlap.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📚 Study Habits")
            study_hours = time_input_combined("Study", 12, 5, "study")
            smartphone_usage = time_input_combined("Phone usage", 10, 3, "phone")
            social_media = time_input_combined("Social media", 8, 2, "social")
            
            st.markdown("#### 🎮 Entertainment (Can Overlap)")
            gaming = time_input_combined("Gaming", 8, 1, "gaming")
            streaming = time_input_combined("Streaming", 8, 1.5, "streaming")
            st.caption("💡 Gaming and Streaming can happen at the same time")
        
        with col2:
            st.markdown("#### 💪 Health & Wellness")
            sleep = time_input_combined("Sleep", 10, 7, "sleep")
            exercise = time_input_combined("Exercise", 3, 1, "exercise")
            caffeine = st.slider("Caffeine cups per day", 0, 8, 1)
            
            st.markdown("#### 📊 Academic")
            attendance = st.slider("Class attendance %", 0, 100, 85)
            assignments = st.slider("Assignment completion %", 0, 100, 80)
            motivation = st.slider("Motivation level (1-10)", 1, 10, 7)
        
        # Demographics
        st.markdown("#### 👤 Demographics")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            mental_health = st.selectbox("Mental health", ["Good", "Average", "Poor"])
        with col3:
            internet = st.selectbox("Internet quality", ["Good", "Average", "Poor"])
        
        # ============================================================
        # VALIDATION
        # ============================================================
        max_entertainment = max(gaming, streaming)
        total_hours = study_hours + smartphone_usage + social_media + sleep + exercise + max_entertainment
        
        col1, col2 = st.columns(2)
        
        with col1:
            if total_hours > 24:
                st.error(f"⚠️ **Total exceeds 24h!** Current: {format_hours(total_hours)}")
                st.markdown(f"""
                <div style="background: #fee; padding: 0.8rem; border-radius: 8px; border: 1px solid #e74c3c;">
                    <b>⏱️ Breakdown:</b><br>
                    Study: {format_hours(study_hours)}<br>
                    Phone: {format_hours(smartphone_usage)}<br>
                    Social Media: {format_hours(social_media)}<br>
                    Sleep: {format_hours(sleep)}<br>
                    Exercise: {format_hours(exercise)}<br>
                    Entertainment: {format_hours(max_entertainment)}<br>
                    <b>Total: {format_hours(total_hours)} / 24h</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"✅ **Valid schedule!** Total: {format_hours(total_hours)} / 24h")
                st.caption(f"Remaining: {format_hours(24 - total_hours)}")
        
        with col2:
            st.markdown("""
            <div style="background: #f0f2f6; padding: 0.8rem; border-radius: 8px;">
                <b>📊 Time Breakdown:</b><br>
            """, unsafe_allow_html=True)
            st.caption(f"📚 Study: {format_hours(study_hours)}")
            st.caption(f"📱 Phone: {format_hours(smartphone_usage)}")
            st.caption(f"💬 Social Media: {format_hours(social_media)}")
            st.caption(f"😴 Sleep: {format_hours(sleep)}")
            st.caption(f"🏃 Exercise: {format_hours(exercise)}")
            if gaming > 0 and streaming > 0:
                st.caption(f"🎮 Gaming: {format_hours(gaming)} | 📺 Streaming: {format_hours(streaming)}")
                st.caption(f"⏱️ Effective Entertainment: {format_hours(max_entertainment)} (overlap counted once)")
            else:
                st.caption(f"🎮 Gaming: {format_hours(gaming)}")
                st.caption(f"📺 Streaming: {format_hours(streaming)}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Create student data dictionary
        student_data = {
            'study_hours_per_day': study_hours,
            'smartphone_usage_hours': smartphone_usage,
            'social_media_hours': social_media,
            'gaming_hours': gaming,
            'streaming_hours': streaming,
            'sleep_hours': sleep,
            'exercise_hours': exercise,
            'caffeine_intake_cups': caffeine,
            'class_attendance_percent': attendance,
            'assignment_completion_percent': assignments,
            'motivation_level': motivation,
            'gender': gender,
            'mental_health_status': mental_health,
            'internet_quality': internet
        }
        
        # Predict button
        if total_hours > 24:
            st.warning("⚠️ Please adjust hours to be within 24 hours.")
            st.button("🔮 Predict Score!", use_container_width=True, disabled=True)
        else:
            if st.button("🔮 Predict Score!", use_container_width=True):
                with st.spinner("Analyzing student data..."):
                    score = predict_score(student_data)
                    
                    if score is not None:
                        # Add points
                        st.session_state.predictions_made += 1
                        
                        # Display score
                        st.markdown("---")
                        
                        # Determine grade
                        if score >= 85:
                            grade = "🌟 EXCELLENT!"
                        elif score >= 70:
                            grade = "👍 GOOD!"
                        elif score >= 60:
                            grade = "📘 SATISFACTORY"
                        else:
                            grade = "⚠️ NEEDS IMPROVEMENT"
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown(f"""
                            <div class="score-card">
                                <h2>🎯 Predicted Score</h2>
                                <div class="score-number">{score:.0f}</div>
                                <div class="grade-badge">{grade}</div>
                                <p style="margin-top: 1rem; opacity: 0.8;">+1 Prediction!</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Prediction failed. Please try again.")

# ============================================================
# MAIN
# ============================================================
def main():
    choice = show_sidebar()
    
    if choice == "🏠 Home":
        home_page()
    elif choice == "📊 Predict":
        prediction_page()

if __name__ == "__main__":
    main()
