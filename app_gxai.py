# app_model2_complete.py - Complete Streamlit Frontend for Gamified Model 2
"""
🎮 SMART STUDENT SCORE PREDICTOR - COMPLETE GAMIFIED STREAMLIT APP
All features from the terminal version converted to web interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from datetime import datetime
import random
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🎮 Student Score Predictor - Gamified",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
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
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .achievement-popup {
            animation: slideIn 0.5s;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-weight: bold;
        }
        @keyframes slideIn {
            0% { transform: translateY(-100px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
        .level-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
            color: #333;
            font-weight: bold;
        }
        .points-display {
            font-size: 1.2rem;
            font-weight: bold;
            color: #f39c12;
        }
        .challenge-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        .quiz-option {
            padding: 0.8rem;
            margin: 0.3rem 0;
            border-radius: 8px;
            background: #f8f9fa;
            cursor: pointer;
            transition: all 0.3s;
            border: 2px solid transparent;
        }
        .quiz-option:hover {
            background: #e9ecef;
        }
        .quiz-correct {
            border-color: #2ecc71 !important;
            background: #d5f5e3 !important;
        }
        .quiz-incorrect {
            border-color: #e74c3c !important;
            background: #fadbd8 !important;
        }
        .progress-bar-container {
            background: #f0f2f6;
            border-radius: 10px;
            padding: 0.2rem;
            margin: 0.5rem 0;
        }
        .progress-bar-fill {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            height: 20px;
            transition: width 0.5s;
        }
        .hint-box {
            background: #f8f9fa;
            padding: 0.8rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin: 0.5rem 0;
            font-size: 0.9rem;
        }
        .challenge-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Initialize all session state variables"""
    # Game state
    if 'game_score' not in st.session_state:
        st.session_state.game_score = 0
    if 'streak' not in st.session_state:
        st.session_state.streak = 0
    if 'best_streak' not in st.session_state:
        st.session_state.best_streak = 0
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = 0
    if 'achievements' not in st.session_state:
        st.session_state.achievements = []
    if 'level' not in st.session_state:
        st.session_state.level = 1
    if 'challenges_completed' not in st.session_state:
        st.session_state.challenges_completed = 0
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = 0
    if 'features_compared' not in st.session_state:
        st.session_state.features_compared = []
    if 'game_history' not in st.session_state:
        st.session_state.game_history = []
    
    # Quiz state
    if 'quiz_active' not in st.session_state:
        st.session_state.quiz_active = False
    if 'quiz_question_idx' not in st.session_state:
        st.session_state.quiz_question_idx = 0
    if 'quiz_score_temp' not in st.session_state:
        st.session_state.quiz_score_temp = 0
    if 'quiz_answered' not in st.session_state:
        st.session_state.quiz_answered = False
    if 'quiz_selected_answer' not in st.session_state:
        st.session_state.quiz_selected_answer = None
    if 'quiz_asked_questions' not in st.session_state:
        st.session_state.quiz_asked_questions = []
    
    # Challenge mode state
    if 'challenge_active' not in st.session_state:
        st.session_state.challenge_active = False
    if 'challenge_completed_list' not in st.session_state:
        st.session_state.challenge_completed_list = []
    if 'current_challenge' not in st.session_state:
        st.session_state.current_challenge = None
    if 'show_feature_hint' not in st.session_state:
        st.session_state.show_feature_hint = False
    
    # Student comparison state
    if 'comparison_active' not in st.session_state:
        st.session_state.comparison_active = False

init_session_state()

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================
@st.cache_resource
def load_model_and_scaler():
    """Load Model 2, scaler, and feature columns"""
    
    model_path = os.path.join(BASE_DIR, "final_exam_model1.pkl")
    scaler_path = os.path.join(BASE_DIR, "robust_scaler1.pkl")
    columns_path = os.path.join(BASE_DIR, "model_columns1.pkl")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        return None, None, None
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(columns_path)
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_dataset():
    """Load the student dataset"""
    dataset_path = os.path.join(BASE_DIR, "student_digital_life.csv")
    
    if not os.path.exists(dataset_path):
        return None
    
    try:
        df = pd.read_csv(dataset_path)
        return df
    except Exception as e:
        return None

model, scaler, feature_columns = load_model_and_scaler()
df_full = load_dataset()

# Create SHAP explainer
if model is not None:
    try:
        explainer = shap.TreeExplainer(model)
        SHAP_AVAILABLE = True
    except:
        explainer = None
        SHAP_AVAILABLE = False
else:
    explainer = None
    SHAP_AVAILABLE = False

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def format_hours(hours):
    """Format hours as hours and minutes"""
    hrs = int(hours)
    # Use round() to fix floating-point errors
    mins = int(round((hours - hrs) * 60))
    
    # If minutes is 60, add to hours and reset minutes
    if mins >= 60:
        hrs += 1
        mins = 0
        
    if hrs > 0 and mins > 0:
        return f"{hrs}h {mins}m"
    elif hrs > 0:
        return f"{hrs}h {mins}m" if mins > 0 else f"{hrs}h"
    elif mins > 0:
        return f"{mins}m"
    else:
        return "0m"

def apply_feature_engineering(df):
    """Apply all feature engineering"""
    df = df.copy()
    
    df["study_efficiency"] = np.clip(
        df["study_hours_per_day"] / (df["sleep_hours"] + 1e-5), 0, 10
    )      
    
    screen_cols = ["social_media_hours", "gaming_hours", "streaming_hours"]
    df["total_screen_time"] = df[screen_cols].sum(axis=1)
    df["distraction_ratio"] = df["total_screen_time"] / (df["study_hours_per_day"] + 1e-5)
    df["health_score"] = df["sleep_hours"] + df["exercise_hours"]
    df["stress_proxy"] = df["caffeine_intake_cups"] / (df["sleep_hours"] + 1e-5)
    df["engagement_score"] = df[["class_attendance_percent", "assignment_completion_percent"]].mean(axis=1)
    df["balance_score"] = df["study_hours_per_day"] / (df["total_screen_time"] + df["sleep_hours"] + 1e-5)
    df["motivation_effect"] = df["motivation_level"] * df["study_hours_per_day"]
    df["study_vs_screen"] = df["study_hours_per_day"] / (df["total_screen_time"] + 1e-5)
    df["sleep_efficiency"] = df["sleep_hours"] * df["study_efficiency"]
    df["mental_pressure"] = df["stress_proxy"] * df["distraction_ratio"]
    df["screen_per_study_hour"] = df["total_screen_time"] / (df["study_hours_per_day"] + 1e-5)
    
    return df

def prepare_prediction(student):
    """Prepare data for model prediction"""
    df = pd.DataFrame([student])
    df = apply_feature_engineering(df)
    df = pd.get_dummies(df, drop_first=True)
    
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_columns]
    df_scaled = scaler.transform(df)
    return df_scaled

def predict_score(student_data):
    """Make prediction"""
    if model is None:
        return None
    
    try:
        features = prepare_prediction(student_data)
        prediction = model.predict(features)[0]
        return np.clip(prediction, 0, 100)
    except Exception as e:
        return None

def add_points(points, reason=""):
    """Add points and check achievements"""
    st.session_state.game_score += points
    st.session_state.predictions_made += 1
    
    # Check level up
    new_level = st.session_state.game_score // 200 + 1
    if new_level > st.session_state.level:
        st.session_state.level = new_level
        st.balloons()
        st.success(f"🎉 LEVEL UP! You're now Level {new_level}!")
        add_achievement("💎 Level Up!")
    
    # Log history
    st.session_state.game_history.append({
        'time': datetime.now().strftime("%H:%M:%S"),
        'action': reason,
        'points': points,
        'total': st.session_state.game_score
    })
    
    # Check achievements
    check_achievements()

def add_achievement(achievement):
    """Add achievement if not already unlocked"""
    if achievement not in st.session_state.achievements:
        st.session_state.achievements.append(achievement)
        st.markdown(f"""
        <div class="achievement-popup">
            🏆 Achievement Unlocked: {achievement}!
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

def check_achievements():
    """Check and unlock achievements"""
    achievements = {
        "🎯 First Prediction": st.session_state.predictions_made >= 1,
        "📚 10 Predictions": st.session_state.predictions_made >= 10,
        "🔥 5-Streak Champion": st.session_state.best_streak >= 5,
        "🏆 Challenge Master": st.session_state.challenges_completed >= 3,
        "💎 Level 5 Master": st.session_state.level >= 5,
        "📊 Feature Explorer": len(st.session_state.features_compared) >= 5,
        "🧠 Quiz Master": st.session_state.quiz_score >= 5,
        "👑 Ultimate Predictor": st.session_state.game_score >= 1000
    }
    
    for achievement, condition in achievements.items():
        if condition and achievement not in st.session_state.achievements:
            add_achievement(achievement)

def get_level_title(level):
    """Get title for level"""
    titles = {
        1: "🥉 Bronze Predictor",
        2: "🥈 Silver Analyst",
        3: "🥇 Gold Guru",
        4: "💎 Platinum Professor",
        5: "👑 Diamond Master"
    }
    return titles.get(level, "🌟 Learner")


# ============================================================
# PDP PLOTS GENERATOR - ONLY 1D PLOTS
# ============================================================

def generate_pdp_plots(feature1, feature2):
    """Generate 1D PDP plots for two features"""
    
    if df_full is None or model is None or scaler is None:
        st.warning("⚠️ Required data not available for PDP plots")
        return
    
    try:
        # Sample data
        sample_df = df_full.sample(n=300, random_state=42).copy()
        sample_df = apply_feature_engineering(sample_df)
        original_values = sample_df.copy()
        
        # Prepare data
        sample_encoded = pd.get_dummies(sample_df, drop_first=True)
        for col in feature_columns:
            if col not in sample_encoded.columns:
                sample_encoded[col] = 0
        sample_encoded = sample_encoded[feature_columns]
        
        # Check if features exist
        if feature1 not in feature_columns or feature2 not in feature_columns:
            st.warning(f"⚠️ Features '{feature1}' or '{feature2}' not found in model")
            return
        
        feat1_idx = feature_columns.index(feature1)
        feat2_idx = feature_columns.index(feature2)
        
        feat1_original = original_values[feature1].values if feature1 in original_values.columns else None
        feat2_original = original_values[feature2].values if feature2 in original_values.columns else None
        
        # Create grid using original feature values
        if feat1_original is not None:
            feat1_grid = np.linspace(min(feat1_original), max(feat1_original), 20)
            feat1_display = feature1.replace('_', ' ').title()
        else:
            feat1_grid = np.linspace(-2, 2, 20)
            feat1_display = feature1.replace('_', ' ').title()
        
        if feat2_original is not None:
            feat2_grid = np.linspace(min(feat2_original), max(feat2_original), 20)
            feat2_display = feature2.replace('_', ' ').title()
        else:
            feat2_grid = np.linspace(-2, 2, 20)
            feat2_display = feature2.replace('_', ' ').title()
        
        # Calculate PDP for Feature 1
        pdp1_values = []
        for val in feat1_grid:
            temp_data = sample_encoded.copy()
            temp_data.iloc[:, feat1_idx] = val
            temp_scaled = scaler.transform(temp_data.values)
            preds = model.predict(temp_scaled)
            pdp1_values.append(np.mean(preds))
        
        # Calculate PDP for Feature 2
        pdp2_values = []
        for val in feat2_grid:
            temp_data = sample_encoded.copy()
            temp_data.iloc[:, feat2_idx] = val
            temp_scaled = scaler.transform(temp_data.values)
            preds = model.predict(temp_scaled)
            pdp2_values.append(np.mean(preds))
        
        # FIGURE: Two 1D PDP plots side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot PDP for Feature 1
        if feat1_original is not None:
            axes[0].plot(feat1_grid, pdp1_values, 'b-', linewidth=2.5, label='PDP')
            axes[0].fill_between(feat1_grid, 
                                np.array(pdp1_values) - np.std(pdp1_values)*0.3,
                                np.array(pdp1_values) + np.std(pdp1_values)*0.3,
                                alpha=0.2, color='blue')
        else:
            axes[0].plot(feat1_grid, pdp1_values, 'b-', linewidth=2.5, label='PDP')
        
        axes[0].set_xlabel(feat1_display, fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Average Predicted Score', fontsize=12, fontweight='bold')
        axes[0].set_title(f'1D PDP: {feat1_display}', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=np.mean(pdp1_values), color='red', linestyle='--', alpha=0.5, label='Average')
        axes[0].legend()
        
        # Plot PDP for Feature 2
        if feat2_original is not None:
            axes[1].plot(feat2_grid, pdp2_values, 'g-', linewidth=2.5, label='PDP')
            axes[1].fill_between(feat2_grid,
                                np.array(pdp2_values) - np.std(pdp2_values)*0.3,
                                np.array(pdp2_values) + np.std(pdp2_values)*0.3,
                                alpha=0.2, color='green')
        else:
            axes[1].plot(feat2_grid, pdp2_values, 'g-', linewidth=2.5, label='PDP')
        
        axes[1].set_xlabel(feat2_display, fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Average Predicted Score', fontsize=12, fontweight='bold')
        axes[1].set_title(f'1D PDP: {feat2_display}', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=np.mean(pdp2_values), color='red', linestyle='--', alpha=0.5, label='Average')
        axes[1].legend()
        
        plt.suptitle('1D Partial Dependence Plots (PDP)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show feature ranges
        if feat1_original is not None and feat2_original is not None:
            st.caption(f"📌 {feat1_display}: {min(feat1_original):.2f} to {max(feat1_original):.2f} | {feat2_display}: {min(feat2_original):.2f} to {max(feat2_original):.2f}")
        
    except Exception as e:
        st.error(f"⚠️ Error generating PDP plots: {e}")

# ============================================================
# GAMIFICATION SIDEBAR
# ============================================================
def show_sidebar():
    """Display gamification sidebar"""
    with st.sidebar:
        st.markdown("### 🎮 Game Dashboard")
        
        level_title = get_level_title(st.session_state.level)
        st.markdown(f"**👤 Title:** {level_title}")
        st.markdown(f"**⭐ Points:** {st.session_state.game_score}")
        st.markdown(f"**🔥 Streak:** {st.session_state.streak}")
        st.markdown(f"**📊 Predictions:** {st.session_state.predictions_made}")
        
        # Level progress
        progress = (st.session_state.game_score % 200) / 200
        st.progress(progress, text=f"Level {st.session_state.level} → {200 - (st.session_state.game_score % 200)} pts to next level")
        
        # Achievements
        st.markdown("### 🏅 Achievements")
        if st.session_state.achievements:
            for ach in st.session_state.achievements[:6]:
                st.markdown(f"- {ach}")
            if len(st.session_state.achievements) > 6:
                st.caption(f"+{len(st.session_state.achievements) - 6} more")
        else:
            st.info("Complete challenges to earn achievements!")
        
        st.markdown("---")
        
        # Navigation
        menu = ["🏠 Home", "📊 Predict", "🎯 Challenges", "📈 Progress"]
        choice = st.radio("Navigate", menu, index=0)
        
        if st.button("🔄 Reset Game"):
            for key in ['game_score', 'streak', 'best_streak', 'predictions_made', 
                       'achievements', 'level', 'challenges_completed', 'quiz_score',
                       'features_compared', 'game_history', 'quiz_active', 'quiz_question_idx',
                       'quiz_score_temp', 'quiz_answered', 'quiz_selected_answer', 'quiz_asked_questions']:
                if key in st.session_state:
                    if key in ['achievements', 'game_history', 'features_compared', 'quiz_asked_questions']:
                        st.session_state[key] = []
                    elif key in ['quiz_active', 'quiz_answered']:
                        st.session_state[key] = False
                    elif key in ['quiz_question_idx', 'quiz_score_temp', 'quiz_selected_answer']:
                        st.session_state[key] = None
                    else:
                        st.session_state[key] = 0
            st.rerun()
        
        return choice

# ============================================================
# HOME PAGE
# ============================================================
def home_page():
    st.markdown('<div class="main-header">🎮 Student Score Predictor - Gamified</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3>📊 Make Predictions</h3>
            <p>Enter student data and get AI-powered exam score predictions with full XAI explanations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3>🎯 Complete Challenges</h3>
            <p>Earn points and unlock achievements by completing interactive challenges.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3>🏆 Earn Rewards</h3>
            <p>Level up, earn points, and unlock achievements as you learn about AI predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stats Overview
    st.markdown("### 📊 Your Stats")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⭐ Points", st.session_state.game_score)
    with col2:
        st.metric("🏅 Achievements", len(st.session_state.achievements))
    with col3:
        st.metric("📊 Predictions", st.session_state.predictions_made)
    with col4:
        st.metric("🔥 Best Streak", st.session_state.best_streak)

# ============================================================
# PREDICTION PAGE
# ============================================================
def prediction_page():
    st.markdown('<div class="main-header">📊 Make a Prediction</div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Model not loaded. Please check the model files.")
        return
    
    with st.container():
        st.markdown("### 📋 Student Information")
        st.info("ℹ️ Select time in minutes (5-minute increments). Total cannot exceed 24 hours.")
        
        # Helper to create minute slider
        def time_slider(label, max_minutes, default_minutes, key):
            minutes = st.slider(
                label,
                0, max_minutes, default_minutes, 5,
                key=key,
                format="%d min"
            )
            hours = minutes // 60
            mins = minutes % 60
            if hours > 0 and mins > 0:
                st.caption(f"⏱️ {hours}h {mins}m")
            elif hours > 0:
                st.caption(f"⏱️ {hours}h")
            elif mins > 0:
                st.caption(f"⏱️ {mins}m")
            else:
                st.caption("⏱️ 0m")
            return minutes / 60
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📚 Study Habits")
            study_hours = time_slider("Study hours", 720, 300, "study")
            smartphone_usage = time_slider("Phone usage", 600, 180, "phone")
            social_media = time_slider("Social media", 480, 120, "social")
            
            st.markdown("#### 🎮 Entertainment (Can Overlap)")
            gaming = time_slider("Gaming hours", 480, 60, "gaming")
            streaming = time_slider("Streaming hours", 480, 90, "streaming")
            st.caption("💡 Gaming and Streaming can overlap")
        
        with col2:
            st.markdown("#### 💪 Health & Wellness")
            sleep = time_slider("Sleep hours", 600, 420, "sleep")
            exercise = time_slider("Exercise hours", 180, 60, "exercise")
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
        
        # Calculate total hours
        max_entertainment = max(gaming, streaming)
        total_hours = study_hours + smartphone_usage + social_media + sleep + exercise + max_entertainment
        
        # Display validation status
        if total_hours > 24:
            st.error(f"⚠️ **Total exceeds 24h!** Current: {format_hours(total_hours)} / 24h")
            
            st.markdown("""
            <div style="background: #fee; padding: 0.8rem; border-radius: 8px; border: 1px solid #e74c3c;">
                <b>⏱️ Breakdown:</b><br>
            """, unsafe_allow_html=True)
            st.caption(f"📚 Study: {format_hours(study_hours)}")
            st.caption(f"📱 Phone: {format_hours(smartphone_usage)}")
            st.caption(f"💬 Social Media: {format_hours(social_media)}")
            st.caption(f"😴 Sleep: {format_hours(sleep)}")
            st.caption(f"🏃 Exercise: {format_hours(exercise)}")
            st.caption(f"🎮 Entertainment: {format_hours(max_entertainment)}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.warning("⚠️ Please adjust hours to be within 24 hours.")
            st.button("🔮 Predict Score!", disabled=True)
        else:
            st.success(f"✅ **Valid schedule!** Total: {format_hours(total_hours)} / 24h")
            st.caption(f"Remaining: {format_hours(24 - total_hours)}")
            
            st.markdown("""
            <div style="background: #f0f2f6; padding: 0.8rem; border-radius: 8px;">
                <b>📊 Time Breakdown:</b><br>
            """, unsafe_allow_html=True)
            st.caption(f"📚 Study: {format_hours(study_hours)}")
            st.caption(f"📱 Phone: {format_hours(smartphone_usage)}")
            st.caption(f"💬 Social Media: {format_hours(social_media)}")
            st.caption(f"😴 Sleep: {format_hours(sleep)}")
            st.caption(f"🏃 Exercise: {format_hours(exercise)}")
            st.caption(f"🎮 Entertainment: {format_hours(max_entertainment)}")
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
            
            if st.button("🔮 Predict Score!"):
                with st.spinner("Analyzing student data..."):
                    score = predict_score(student_data)
                    
                    if score is not None:
                        # Add points
                        points = 10
                        if score >= 90: points += 30
                        elif score >= 75: points += 20
                        elif score >= 60: points += 10
                        
                        if st.session_state.streak > 0:
                            points += min(50, st.session_state.streak * 2)
                        
                        add_points(points, f"Prediction: {score:.0f}/100")
                        
                        # Update streak
                        if score >= 60:
                            st.session_state.streak += 1
                            if st.session_state.streak > st.session_state.best_streak:
                                st.session_state.best_streak = st.session_state.streak
                        else:
                            st.session_state.streak = 0
                        
                        # Display score
                        st.markdown("---")
                        
                        if score >= 90:
                            grade = "🌟 A+ (Excellent)"
                        elif score >= 80:
                            grade = "⭐ A (Very Good)"
                        elif score >= 70:
                            grade = "✅ B (Good)"
                        elif score >= 60:
                            grade = "📖 C (Average)"
                        elif score >= 50:
                            grade = "⚠️ D (Below Average)"
                        else:
                            grade = "❌ F (Needs Improvement)"
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.markdown(f"""
                            <div class="score-card">
                                <h2>🎯 Predicted Score</h2>
                                <div class="score-number">{score:.0f}</div>
                                <div class="grade-badge">{grade}</div>
                                <p style="margin-top: 1rem; opacity: 0.8;">+{points} points earned!</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # SHAP Explanation Only (LIME removed)
                        st.markdown("---")
                        st.markdown("### 🔍 Understand Your Prediction with SHAP")
                        generate_shap_explanation(student_data, score)
                        
                        # Recommendations
                        st.markdown("---")
                        st.markdown("### 💡 Personalized Recommendations")
                        generate_recommendations(student_data, score)
                    else:
                        st.error("❌ Prediction failed. Please try again.")

# ============================================================
# SHAP EXPLANATION
# ============================================================
def generate_shap_explanation(student_data, score):
    """Generate SHAP explanation visualizations"""
    
    if not SHAP_AVAILABLE:
        st.warning("⚠️ SHAP not available")
        return
    
    try:
        features = prepare_prediction(student_data)
        X_input = pd.DataFrame(features, columns=feature_columns)
        shap_values = explainer.shap_values(X_input)
        
        shap_contrib = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray):
            base_val = base_val[0]
        
        # Waterfall Plot
        st.markdown("#### SHAP Waterfall Plot")
        st.markdown("*Shows how each feature contributed to this prediction*")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_contrib,
                base_values=base_val,
                data=X_input.iloc[0].values,
                feature_names=feature_columns
            ),
            show=False,
            max_display=15
        )
        plt.title(f'SHAP Waterfall Plot\nPredicted Score: {score:.0f}/100', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Feature Importance
        st.markdown("#### SHAP Feature Importance")
        st.markdown("*Features with the highest impact on the prediction*")
        
        shap_df = pd.DataFrame({
            'Feature': feature_columns,
            'Contribution': shap_contrib
        }).sort_values('Contribution', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in shap_df['Contribution']]
        plt.barh(shap_df['Feature'][:10][::-1], shap_df['Contribution'][:10][::-1], 
                 color=colors[:10][::-1], alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
        plt.xlabel('SHAP Contribution', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title(f'Top 10 Features - SHAP Contributions\nScore: {score:.0f}/100', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        add_points(15, "Viewed SHAP explanation")
        st.info("📊 +15 points for exploring SHAP!")
        
    except Exception as e:
        st.error(f"⚠️ SHAP analysis failed: {e}")

# ============================================================
# RECOMMENDATIONS
# ============================================================
def generate_recommendations(student_data, score):
    """Generate personalized recommendations"""
    
    recommendations = []
    
    study = student_data.get('study_hours_per_day', 0)
    if study < 4:
        recommendations.append("📚 Increase study hours to 4-6 hours daily for better performance")
    elif study < 6:
        recommendations.append("📚 Good study hours! Try to maintain 6+ hours for excellent results")
    
    sleep = student_data.get('sleep_hours', 0)
    if sleep < 6:
        recommendations.append("😴 Improve sleep to 7-8 hours - sleep is crucial for memory and focus")
    elif sleep < 7:
        recommendations.append("😴 Good sleep! Try to reach 8 hours for optimal brain function")
    
    screen_total = (student_data.get('social_media_hours', 0) + 
                   student_data.get('gaming_hours', 0) + 
                   student_data.get('streaming_hours', 0))
    if screen_total > 5:
        recommendations.append("📱 Reduce screen time to under 4 hours - excessive screen time hurts focus")
    
    attendance = student_data.get('class_attendance_percent', 0)
    if attendance < 75:
        recommendations.append("📖 Improve attendance to 85%+ - every missed class impacts learning")
    
    assignments = student_data.get('assignment_completion_percent', 0)
    if assignments < 70:
        recommendations.append("✏️ Complete more assignments - they are crucial for exam preparation")
    
    motivation = student_data.get('motivation_level', 0)
    if motivation < 5:
        recommendations.append("💪 Find ways to boost motivation - set small achievable goals")
    
    if score >= 75:
        recommendations.append("🌟 Great work! Keep maintaining your good habits!")
    
    if recommendations:
        for rec in recommendations[:5]:
            st.success(rec)
    else:
        st.info("📚 You're on the right track! Keep up the good habits!")

# ============================================================
# FEATURE IMPORTANCE HINT - DISPLAY IN DROPDOWN
# ============================================================
def show_feature_importance_hint():
    """Display feature importance chart in an expandable dropdown"""
    
    if df_full is None:
        st.warning("⚠️ Dataset not available")
        return
    
    with st.expander("📊 View Feature Importance Chart (Hint)", expanded=False):
        st.markdown("""
        <div class="hint-box">
            💡 <b>Understanding Feature Importance:</b> This chart shows which features have the strongest impact on exam scores.
            The longer the bar, the more important the feature. Use this to guide your challenge strategy!
        </div>
        """, unsafe_allow_html=True)
        
        try:
            sample_df = df_full.sample(n=200, random_state=42).copy()
            sample_df = apply_feature_engineering(sample_df)
            
            sample_encoded = pd.get_dummies(sample_df, drop_first=True)
            for col in feature_columns:
                if col not in sample_encoded.columns:
                    sample_encoded[col] = 0
            sample_encoded = sample_encoded[feature_columns]
            X_sample = sample_encoded.values
            y_sample = sample_df['final_exam_score'].values[:len(X_sample)]
            X_sample_scaled = scaler.transform(X_sample)
            
            try:
                # Try permutation importance first
                perm_importance = permutation_importance(
                    model, X_sample_scaled, y_sample,
                    n_repeats=5, random_state=42, scoring='r2'
                )
                
                perm_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': perm_importance.importances_mean,
                    'Std': perm_importance.importances_std
                }).sort_values('Importance', ascending=False).head(15)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(perm_df)))[::-1]
                bars = plt.barh(perm_df['Feature'][::-1], perm_df['Importance'][::-1],
                               color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add value labels
                for bar, val in zip(bars, perm_df['Importance'][::-1]):
                    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                            f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
                
                ax.set_xlabel('Permutation Importance (R² decrease)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Features', fontsize=11, fontweight='bold')
                ax.set_title('Feature Importance - Most Impactful Features', fontsize=13, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.caption("📌 **Top 3 Most Important Features:**")
                top_features = perm_df.head(3)['Feature'].tolist()
                for i, f in enumerate(top_features, 1):
                    st.markdown(f"- **{i}. {f.replace('_', ' ').title()}**")
                    
            except:
                # Fallback to model feature importances
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(importance_df)))[::-1]
                plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                plt.xlabel('Feature Importance', fontsize=11, fontweight='bold')
                plt.ylabel('Features', fontsize=11, fontweight='bold')
                plt.title('Feature Importance - Most Impactful Features', fontsize=13, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                st.caption("📌 **Top 3 Most Important Features:**")
                top_features = importance_df.head(3)['Feature'].tolist()
                for i, f in enumerate(top_features, 1):
                    st.markdown(f"- **{i}. {f.replace('_', ' ').title()}**")
            
        except Exception as e:
            st.warning(f"⚠️ Could not generate feature importance chart: {e}")

# ============================================================
# CHALLENGES PAGE
# ============================================================
def challenges_page():
    """Challenges page with menu system - each challenge on its own page"""
    
    st.markdown('<div class="main-header">🎯 Challenges Hub</div>', unsafe_allow_html=True)
    
    # Initialize challenge selection state
    if 'selected_challenge' not in st.session_state:
        st.session_state.selected_challenge = None
    
    # Show progress summary at top
    st.markdown("""
    ### Welcome to the Challenge Hub!
    Select a challenge below to test your skills and earn points.
    """)
    
    # Progress Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Challenges Completed", st.session_state.challenges_completed)
    with col2:
        st.metric("🧠 Quiz Score", st.session_state.quiz_score)
    with col3:
        st.metric("📊 Features Compared", len(st.session_state.features_compared))
    with col4:
        st.metric("⭐ Points Earned", st.session_state.game_score)
    
    st.markdown("---")
    
    # Challenge Menu
    if st.session_state.selected_challenge is not None:
        # Back button
        if st.button("⬅️ Back to Challenges Menu"):
            st.session_state.selected_challenge = None
            st.rerun()
        
        st.markdown("---")
        
        # Show the selected challenge
        if st.session_state.selected_challenge == "PDP Quiz":
            show_pdp_quiz_page()
        elif st.session_state.selected_challenge == "Compare Features":
            show_feature_comparison_page()
        elif st.session_state.selected_challenge == "Compare Students":
            show_student_comparison_page()
        elif st.session_state.selected_challenge == "Challenge Mode":
            show_challenge_mode_page()
        elif st.session_state.selected_challenge == "Global Analysis":
            show_global_analysis_page()
        elif st.session_state.selected_challenge == "Feature Importance":
            show_feature_importance_page()
        return
    
    # Challenge Menu Grid
    col1, col2 = st.columns(2)
    
    with col1:
        # Challenge 1: PDP Quiz
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <h3>🧠 PDP Plots Quiz</h3>
            <p>Test your understanding of Partial Dependence Plots with interactive visualizations and 5 questions.</p>
            <p style="color: #666; font-size: 0.9rem;">🏆 +30 points per correct answer</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 Start PDP Quiz", key="btn_quiz"):
            st.session_state.selected_challenge = "PDP Quiz"
            st.rerun()
        
        # Challenge 2: Compare Features
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <h3>📊 Compare Features</h3>
            <p>Explore how two features interact using SHAP dependence plots with original values.</p>
            <p style="color: #666; font-size: 0.9rem;">🏆 +20 points per comparison</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔬 Compare Features", key="btn_features"):
            st.session_state.selected_challenge = "Compare Features"
            st.rerun()
        
        # Challenge 3: Compare Students
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <h3>👥 Compare Students</h3>
            <p>Compare two student profiles side by side with SHAP waterfall plots and detailed analysis.</p>
            <p style="color: #666; font-size: 0.9rem;">🏆 +30 points per comparison</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👥 Compare Students", key="btn_students"):
            st.session_state.selected_challenge = "Compare Students"
            st.rerun()
    
    with col2:
        # Challenge 4: Challenge Mode
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <h3>🎯 Challenge Mode</h3>
            <p>Complete specific challenges with targets like "The Overachiever" or "The Balanced Student".</p>
            <p style="color: #666; font-size: 0.9rem;">🏆 +100+ bonus points</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start Challenge", key="btn_challenge"):
            st.session_state.selected_challenge = "Challenge Mode"
            st.rerun()
        
        # Challenge 5: Global Analysis
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <h3>🌍 Global Analysis</h3>
            <p>View global feature impact distribution across all students using SHAP Beeswarm plot.</p>
            <p style="color: #666; font-size: 0.9rem;">🏆 +25 points</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🌍 Generate Global Analysis", key="btn_global"):
            st.session_state.selected_challenge = "Global Analysis"
            st.rerun()
        
        # Challenge 6: Feature Importance
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 1rem;">
            <h3>📊 Feature Importance</h3>
            <p>View permutation importance of all features to understand what matters most.</p>
            <p style="color: #666; font-size: 0.9rem;">🏆 +20 points</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📊 Show Feature Importance", key="btn_importance"):
            st.session_state.selected_challenge = "Feature Importance"
            st.rerun()

# ============================================================
# PDP QUIZ PAGE
# ============================================================
def show_pdp_quiz_page():
    """Dedicated PDP Quiz page with only 1D PDP plots"""
    
    st.markdown("### 🧠 PDP Plots Quiz")
    st.markdown("*Test your understanding of 1D Partial Dependence Plots with interactive visualizations*")
    st.markdown(f"**Your Score:** {st.session_state.quiz_score} correct answers")
    
    st.markdown("---")
    
    # All questions
    all_questions = [
        {"feature1": "study_hours_per_day", "feature2": "sleep_hours",
         "question": "Based on the 1D PDP plots, which feature has a STRONGER positive impact on exam scores?",
         "options": ["A) Study Hours", "B) Sleep Hours", "C) Both have equal impact", "D) Neither has impact"],
         "correct": "A", 
         "explanation": "Study hours show a steeper upward trend, indicating a stronger positive impact on exam scores."},
        {"feature1": "study_hours_per_day", "feature2": "gaming_hours",
         "question": "What pattern do you observe in the 1D PDP plot for gaming hours?",
         "options": ["A) Scores increase with gaming", "B) Scores decrease with gaming", "C) No clear pattern", "D) Scores fluctuate wildly"],
         "correct": "B", 
         "explanation": "The 1D PDP plot for gaming hours typically shows a downward trend, indicating that more gaming hours lead to lower exam scores."},
        {"feature1": "smartphone_usage_hours", "feature2": "study_hours_per_day",
         "question": "Which feature shows a CLEARER upward trend in the 1D PDP plot?",
         "options": ["A) Smartphone Usage", "B) Study Hours", "C) Both show upward trends", "D) Neither shows clear trend"],
         "correct": "B", 
         "explanation": "Study hours typically show a more pronounced upward trend in the 1D PDP plot compared to smartphone usage."},
        {"feature1": "motivation_level", "feature2": "study_hours_per_day",
         "question": "Which feature shows the STEEPEST increase in the 1D PDP plot?",
         "options": ["A) Motivation Level", "B) Study Hours", "C) Both are similar", "D) Neither shows increase"],
         "correct": "B", 
         "explanation": "Study hours typically show the steepest increase in the 1D PDP plot, indicating the strongest impact on exam scores."},
        {"feature1": "sleep_hours", "feature2": "motivation_level",
         "question": "What is the shape of the 1D PDP plot for sleep hours?",
         "options": ["A) Steep upward", "B) Flat (no change)", "C) Slight upward then plateau", "D) Downward"],
         "correct": "C", 
         "explanation": "The 1D PDP plot for sleep hours typically shows scores increasing up to 7-8 hours, then plateauing."}
    ]
    
    # Initialize quiz state
    if 'quiz_current_index' not in st.session_state:
        st.session_state.quiz_current_index = 0
        st.session_state.quiz_answered = False
        st.session_state.quiz_selected = None
        st.session_state.quiz_temp_score = 0
        st.session_state.quiz_asked_indices = []
        st.session_state.quiz_shuffled = []
    
    # Start new quiz
    if len(st.session_state.quiz_shuffled) == 0:
        st.session_state.quiz_shuffled = all_questions.copy()
        random.shuffle(st.session_state.quiz_shuffled)
        st.session_state.quiz_current_index = 0
        st.session_state.quiz_answered = False
        st.session_state.quiz_selected = None
        st.session_state.quiz_temp_score = 0
        st.session_state.quiz_asked_indices = []
    
    questions = st.session_state.quiz_shuffled
    
    # Check if all questions answered
    if len(st.session_state.quiz_asked_indices) >= len(questions):
        st.balloons()
        st.success(f"🎉 Quiz Complete! Final Score: {st.session_state.quiz_temp_score} / {len(questions)}")
        
        if st.session_state.quiz_temp_score >= 4:
            add_achievement("🧠 Quiz Master")
            st.success("🏆 Achievement Unlocked: Quiz Master!")
        
        if st.button("🔄 Start New Quiz"):
            st.session_state.quiz_shuffled = []
            st.session_state.quiz_current_index = 0
            st.session_state.quiz_answered = False
            st.session_state.quiz_selected = None
            st.session_state.quiz_temp_score = 0
            st.session_state.quiz_asked_indices = []
            st.rerun()
        return
    
    # Get current question
    current_idx = st.session_state.quiz_current_index
    q = questions[current_idx]
    
    # Display progress
    st.markdown(f"**Question {len(st.session_state.quiz_asked_indices) + 1} of {len(questions)}**")
    st.markdown(f"**Score:** {st.session_state.quiz_temp_score} / {len(questions)}")
    
    # Generate 1D PDP plots
    if df_full is not None:
        try:
            generate_pdp_plots(q['feature1'], q['feature2'])
        except Exception as e:
            st.warning(f"⚠️ Could not generate PDP plots: {e}")
    
    # Display question
    st.markdown(f"### ❓ {q['question']}")
    st.markdown("🔍 **HINT:** Look at the trends in the 1D PDP plots above!")
    
    # Display options
    cols = st.columns(2)
    correct_option = next(o for o in q['options'] if o.startswith(q['correct']))
    
    for i, option in enumerate(q['options']):
        col_idx = i % 2
        with cols[col_idx]:
            if st.session_state.quiz_answered:
                if option == correct_option:
                    st.success(f"✅ {option}")
                elif option == st.session_state.quiz_selected:
                    st.error(f"❌ {option}")
                else:
                    st.info(option)
            else:
                if st.button(option, key=f"pdp_opt_{current_idx}_{i}"):
                    st.session_state.quiz_selected = option
                    st.session_state.quiz_answered = True
                    
                    if option == correct_option:
                        st.session_state.quiz_temp_score += 1
                        st.session_state.quiz_score += 1
                        add_points(30, f"Quiz correct! Q{len(st.session_state.quiz_asked_indices) + 1}")
                        st.success(f"✅ Correct! {q['explanation']}")
                    else:
                        st.error(f"❌ Incorrect. The correct answer was {correct_option}")
                        st.info(f"📚 {q['explanation']}")
                        add_points(10, f"Quiz attempt: Q{len(st.session_state.quiz_asked_indices) + 1}")
                    
                    st.session_state.quiz_asked_indices.append(current_idx)
                    st.rerun()
    
    # Next question button
    if st.session_state.quiz_answered:
        if len(st.session_state.quiz_asked_indices) < len(questions):
            if st.button("➡️ Next Question"):
                next_idx = None
                for i in range(len(questions)):
                    if i not in st.session_state.quiz_asked_indices:
                        next_idx = i
                        break
                
                if next_idx is not None:
                    st.session_state.quiz_current_index = next_idx
                    st.session_state.quiz_answered = False
                    st.session_state.quiz_selected = None
                    st.rerun()
    
    # Tips
    if st.session_state.quiz_answered:
        st.markdown("---")
        st.markdown("### 📊 1D PDP Interpretation Tips")
        st.markdown("""
        - **Upward trend** = Feature increases exam scores
        - **Downward trend** = Feature decreases exam scores  
        - **Flat line** = Feature has little impact
        - **Steep slope** = Feature has strong impact
        - **Plateau** = Impact levels off after a certain point
        """)

# ============================================================
# FEATURE COMPARISON PAGE
# ============================================================
def show_feature_comparison_page():
    """Dedicated Feature Comparison page"""
    
    st.markdown("### 📊 Compare Features with SHAP")
    st.markdown("*Explore how two features interact and affect predictions using SHAP*")
    st.markdown(f"**Features Compared:** {len(st.session_state.features_compared)}")
    
    st.markdown("---")
    
    if not SHAP_AVAILABLE:
        st.warning("⚠️ SHAP not available")
        return
    
    if df_full is None:
        st.warning("⚠️ Dataset not available")
        return
    
    # Get available features
    available_features = [f for f in feature_columns if not f.startswith('gender_') and not f.startswith('mental_') and not f.startswith('internet_')]
    if not available_features:
        available_features = feature_columns[:20]
    
    col1, col2 = st.columns(2)
    
    with col1:
        feat1 = st.selectbox("Select first feature:", available_features, key="feat1_compare")
    with col2:
        feat2 = st.selectbox("Select second feature:", available_features, key="feat2_compare")
    
    if feat1 != feat2:
        if st.button("📈 Generate SHAP Dependence Plot"):
            with st.spinner("Generating SHAP dependence plot..."):
                try:
                    sample_df = df_full.sample(n=300, random_state=42).copy()
                    sample_df = apply_feature_engineering(sample_df)
                    
                    sample_encoded = pd.get_dummies(sample_df, drop_first=True)
                    for col in feature_columns:
                        if col not in sample_encoded.columns:
                            sample_encoded[col] = 0
                    sample_encoded = sample_encoded[feature_columns]
                    X_sample = sample_encoded.values
                    X_sample_scaled = scaler.transform(X_sample)
                    
                    shap_values_sample = explainer.shap_values(X_sample_scaled)
                    if isinstance(shap_values_sample, list):
                        shap_values_sample = shap_values_sample[0]
                    
                    feat1_idx = feature_columns.index(feat1)
                    feat2_idx = feature_columns.index(feat2)
                    
                    if feat1 in sample_df.columns:
                        x_vals = sample_df[feat1].values
                    else:
                        x_vals = X_sample_scaled[:, feat1_idx]
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    scatter = ax.scatter(
                        x_vals,
                        shap_values_sample[:, feat1_idx],
                        c=X_sample_scaled[:, feat2_idx],
                        cmap='coolwarm',
                        s=60,
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    
                    z = np.polyfit(x_vals, shap_values_sample[:, feat1_idx], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(x_vals), max(x_vals), 100)
                    ax.plot(x_line, p(x_line), 'r--', alpha=0.6, linewidth=2, label='Trend')
                    
                    ax.set_xlabel(feat1.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                    ax.set_ylabel(f'SHAP value for {feat1.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                    ax.set_title(f'SHAP Dependence Plot\n{feat1.replace("_", " ").title()} vs {feat2.replace("_", " ").title()}', 
                                fontsize=14, fontweight='bold')
                    
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(feat2.replace('_', ' ').title(), fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    add_points(20, f"Compared {feat1} vs {feat2}")
                    if feat1 not in st.session_state.features_compared:
                        st.session_state.features_compared.append(feat1)
                    if feat2 not in st.session_state.features_compared:
                        st.session_state.features_compared.append(feat2)
                    
                    st.success("📊 +20 points for exploring feature interactions!")
                    
                except Exception as e:
                    st.error(f"⚠️ Error generating plot: {e}")
    else:
        st.warning("Please select two DIFFERENT features")

# ============================================================
# STUDENT COMPARISON PAGE
# ============================================================
def show_student_comparison_page():
    """Dedicated Student Comparison page"""
    
    st.markdown("### 👥 Compare Two Students")
    st.markdown("*Create two student profiles and compare their predictions*")
    
    st.markdown("---")
    
    st.markdown("#### Student A")
    col1, col2 = st.columns(2)
    
    with col1:
        study_a = st.slider("Study hours (A)", 0.0, 12.0, 5.0, 0.5, key="comp_study_a")
        sleep_a = st.slider("Sleep hours (A)", 4.0, 10.0, 7.0, 0.5, key="comp_sleep_a")
        motivation_a = st.slider("Motivation (A)", 1, 10, 7, key="comp_mot_a")
    
    with col2:
        gaming_a = st.slider("Gaming hours (A)", 0.0, 8.0, 1.0, 0.5, key="comp_gaming_a")
        attendance_a = st.slider("Attendance % (A)", 0, 100, 85, key="comp_att_a")
        assignments_a = st.slider("Assignments % (A)", 0, 100, 80, key="comp_assign_a")
    
    st.markdown("#### Student B")
    col1, col2 = st.columns(2)
    
    with col1:
        study_b = st.slider("Study hours (B)", 0.0, 12.0, 3.0, 0.5, key="comp_study_b")
        sleep_b = st.slider("Sleep hours (B)", 4.0, 10.0, 6.0, 0.5, key="comp_sleep_b")
        motivation_b = st.slider("Motivation (B)", 1, 10, 5, key="comp_mot_b")
    
    with col2:
        gaming_b = st.slider("Gaming hours (B)", 0.0, 8.0, 3.0, 0.5, key="comp_gaming_b")
        attendance_b = st.slider("Attendance % (B)", 0, 100, 70, key="comp_att_b")
        assignments_b = st.slider("Assignments % (B)", 0, 100, 65, key="comp_assign_b")
    
    if st.button("👥 Compare Students"):
        student_a = {
            'study_hours_per_day': study_a,
            'sleep_hours': sleep_a,
            'motivation_level': motivation_a,
            'gaming_hours': gaming_a,
            'class_attendance_percent': attendance_a,
            'assignment_completion_percent': assignments_a,
            'smartphone_usage_hours': 3,
            'social_media_hours': 2,
            'streaming_hours': 1.5,
            'exercise_hours': 1,
            'caffeine_intake_cups': 1,
            'gender': 'Male',
            'mental_health_status': 'Good',
            'internet_quality': 'Good'
        }
        
        student_b = {
            'study_hours_per_day': study_b,
            'sleep_hours': sleep_b,
            'motivation_level': motivation_b,
            'gaming_hours': gaming_b,
            'class_attendance_percent': attendance_b,
            'assignment_completion_percent': assignments_b,
            'smartphone_usage_hours': 3,
            'social_media_hours': 2,
            'streaming_hours': 1.5,
            'exercise_hours': 1,
            'caffeine_intake_cups': 1,
            'gender': 'Male',
            'mental_health_status': 'Good',
            'internet_quality': 'Good'
        }
        
        score_a = predict_score(student_a)
        score_b = predict_score(student_b)
        
        if score_a is not None and score_b is not None:
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center;">
                    <h3>Student A</h3>
                    <h1 style="color: {'#2ecc71' if score_a >= score_b else '#e74c3c'}; font-size: 4rem;">{score_a:.0f}</h1>
                    <p style="font-size: 1.2rem;">/100</p>
                    <p style="font-size: 0.9rem; color: #666;">Study: {study_a}h | Sleep: {sleep_a}h</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center;">
                    <h3>Student B</h3>
                    <h1 style="color: {'#2ecc71' if score_b >= score_a else '#e74c3c'}; font-size: 4rem;">{score_b:.0f}</h1>
                    <p style="font-size: 1.2rem;">/100</p>
                    <p style="font-size: 0.9rem; color: #666;">Study: {study_b}h | Sleep: {sleep_b}h</p>
                </div>
                """, unsafe_allow_html=True)
            
            winner = "A" if score_a > score_b else "B"
            difference = abs(score_a - score_b)
            st.success(f"🏆 Student {winner} wins by {difference:.0f} points!")
            
            # Comparison table
            st.markdown("---")
            st.markdown("#### 📊 Comparison Table")
            
            comparison_data = {
                'Metric': ['Study Hours', 'Sleep Hours', 'Motivation', 'Gaming Hours', 'Attendance', 'Assignments', 'Predicted Score'],
                'Student A': [f"{study_a}h", f"{sleep_a}h", f"{motivation_a}/10", f"{gaming_a}h", f"{attendance_a}%", f"{assignments_a}%", f"{score_a:.0f}"],
                'Student B': [f"{study_b}h", f"{sleep_b}h", f"{motivation_b}/10", f"{gaming_b}h", f"{attendance_b}%", f"{assignments_b}%", f"{score_b:.0f}"]
            }
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            add_points(30, f"Compared students: A({score_a:.0f}) vs B({score_b:.0f})")
            st.success("👥 +30 points for student comparison!")
            st.session_state.challenges_completed += 1

# ============================================================
# CHALLENGE MODE PAGE
# ============================================================
def show_challenge_mode_page():
    """Dedicated Challenge Mode page with feature importance dropdown"""
    
    st.markdown("### 🎯 Challenge Mode")
    st.markdown("*Complete specific challenges to earn bonus points*")
    
    # Show feature importance hint in dropdown
    show_feature_importance_hint()
    
    st.markdown("---")
    
    all_challenges = [
        {"name": "🚀 The Comeback Kid", "target": 75, 
         "hint": "Focus on study hours (6+) and reducing screen time",
         "required": {"study_hours_per_day": 6, "total_screen_time": 4}},
        {"name": "💪 The Overachiever", "target": 85,
         "hint": "High study hours (7+), excellent attendance (95%+)",
         "required": {"study_hours_per_day": 7, "class_attendance_percent": 95}},
        {"name": "⚖️ The Balanced Student", "target": 75,
         "hint": "Moderate study (5 hours), good sleep (8 hours)",
         "required": {"study_hours_per_day": 5, "sleep_hours": 8}},
        {"name": "📱 The Digital Native", "target": 65,
         "hint": "Study 5+ hours, screen time under 5 hours",
         "required": {"study_hours_per_day": 5, "total_screen_time": 5}},
        {"name": "🌙 The Night Owl", "target": 70,
         "hint": "Sleep 8 hours, study 5 hours",
         "required": {"study_hours_per_day": 5, "sleep_hours": 8}},
        {"name": "🎮 The Gamer", "target": 72,
         "hint": "Study 6+ hours, gaming under 2 hours",
         "required": {"study_hours_per_day": 6, "gaming_hours": 2}},
        {"name": "🏆 The All-Rounder", "target": 80,
         "hint": "Study 6+, sleep 8+, exercise 1+, motivation 8+",
         "required": {"study_hours_per_day": 6, "sleep_hours": 8, "exercise_hours": 1, "motivation_level": 8}}
    ]
    
    # Get available challenges
    available = [c for c in all_challenges if c['name'] not in st.session_state.challenge_completed_list]
    
    if len(available) == 0:
        st.session_state.challenge_completed_list = []
        available = all_challenges
        st.info("🔄 All challenges completed! Resetting...")
    
    # Select a random challenge
    if st.session_state.current_challenge is None or st.session_state.current_challenge not in [c['name'] for c in available]:
        st.session_state.current_challenge = random.choice(available)['name']
    
    challenge = next(c for c in all_challenges if c['name'] == st.session_state.current_challenge)
    
    st.markdown(f"""
    <div class="challenge-header">
        <h3>📋 {challenge['name']}</h3>
        <p style="margin: 0.5rem 0;">🎯 Target: {challenge['target']}+</p>
        <p style="margin: 0; opacity: 0.9;">💡 {challenge['hint']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        study = st.slider("Study hours", 0.0, 12.0, 5.0, 0.5, key="chall_study")
        sleep = st.slider("Sleep hours", 4.0, 10.0, 7.0, 0.5, key="chall_sleep")
        motivation = st.slider("Motivation level", 1, 10, 7, key="chall_mot")
    
    with col2:
        gaming = st.slider("Gaming hours", 0.0, 8.0, 1.0, 0.5, key="chall_gaming")
        attendance = st.slider("Attendance %", 0, 100, 85, key="chall_att")
        assignments = st.slider("Assignments %", 0, 100, 80, key="chall_assign")
    
    if st.button("🎯 Submit Challenge"):
        student = {
            'study_hours_per_day': study,
            'sleep_hours': sleep,
            'motivation_level': motivation,
            'gaming_hours': gaming,
            'class_attendance_percent': attendance,
            'assignment_completion_percent': assignments,
            'smartphone_usage_hours': 3,
            'social_media_hours': 2,
            'streaming_hours': 1.5,
            'exercise_hours': 1,
            'caffeine_intake_cups': 1,
            'gender': 'Male',
            'mental_health_status': 'Good',
            'internet_quality': 'Good'
        }
        
        score = predict_score(student)
        
        if score is not None:
            total_screen = student['social_media_hours'] + student['gaming_hours'] + student['streaming_hours']
            student['total_screen_time'] = total_screen
            
            requirements_met = []
            for key, value in challenge['required'].items():
                if student.get(key, 0) >= value:
                    requirements_met.append(key)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div class="score-card">
                    <h2>🎯 Challenge Score</h2>
                    <div class="score-number">{score:.0f}</div>
                    <p style="margin-top: 1rem;">Target: {challenge['target']}+</p>
                </div>
                """, unsafe_allow_html=True)
            
            progress = min(100, int((score / challenge['target']) * 100))
            st.markdown(f"**Progress:** {progress}%")
            st.progress(progress / 100)
            
            if score >= challenge['target']:
                st.success(f"🎉 CHALLENGE COMPLETED!")
                bonus = 50 + (len(requirements_met) * 10)
                add_points(100 + bonus, f"Completed challenge: {challenge['name']}")
                st.session_state.challenges_completed += 1
                st.session_state.challenge_completed_list.append(challenge['name'])
                st.balloons()
                
                if st.session_state.challenges_completed >= 3 and "🏆 Challenge Master" not in st.session_state.achievements:
                    add_achievement("🏆 Challenge Master")
            else:
                st.error(f"❌ Challenge failed. Need {challenge['target'] - score:.0f} more points.")
                consolation = 20 + len(requirements_met) * 5
                add_points(consolation, f"Challenge attempt: {challenge['name']}")
            
            if requirements_met:
                st.markdown("#### ✅ Requirements Met:")
                for req in requirements_met:
                    st.markdown(f"- {req.replace('_', ' ').title()}: {student.get(req, 0)}")
            
            st.markdown(f"**Challenges Completed:** {st.session_state.challenges_completed}")

# ============================================================
# GLOBAL ANALYSIS PAGE
# ============================================================
def show_global_analysis_page():
    """Dedicated Global Analysis page"""
    
    st.markdown("### 🌍 Global Analysis - SHAP Beeswarm Plot")
    st.markdown("*Shows global feature impact distribution across all students*")
    
    st.markdown("---")
    
    if not SHAP_AVAILABLE:
        st.warning("⚠️ SHAP not available")
        return
    
    if df_full is None:
        st.warning("⚠️ Dataset not available")
        return
    
    with st.spinner("Generating global SHAP analysis..."):
        try:
            sample_df = df_full.sample(n=200, random_state=42).copy()
            sample_df = apply_feature_engineering(sample_df)
            
            sample_encoded = pd.get_dummies(sample_df, drop_first=True)
            for col in feature_columns:
                if col not in sample_encoded.columns:
                    sample_encoded[col] = 0
            sample_encoded = sample_encoded[feature_columns]
            X_sample = sample_encoded.values
            X_sample_scaled = scaler.transform(X_sample)
            
            shap_values_sample = explainer.shap_values(X_sample_scaled)
            if isinstance(shap_values_sample, list):
                shap_values_sample = shap_values_sample[0]
            
            fig, ax = plt.subplots(figsize=(16, 12))
            shap.summary_plot(
                shap_values_sample, X_sample_scaled,
                feature_names=feature_columns, show=False,
                max_display=len(feature_columns), plot_type="dot", color_bar=True
            )
            plt.title('SHAP Beeswarm Plot - Global Feature Impact Distribution', 
                     fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            add_points(25, "Viewed global SHAP analysis")
            st.success("🌍 +25 points for global analysis!")
            
            shap_abs = np.abs(shap_values_sample).mean(axis=0)
            shap_importance = pd.DataFrame({
                'Feature': feature_columns,
                'SHAP_Value': shap_abs
            }).sort_values('SHAP_Value', ascending=False)
            
            st.markdown("#### 📊 Top 10 Features")
            st.dataframe(shap_importance.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"⚠️ Error generating global analysis: {e}")

# ============================================================
# FEATURE IMPORTANCE PAGE
# ============================================================
def show_feature_importance_page():
    """Dedicated Feature Importance page"""
    
    st.markdown("### 📊 Feature Importance - Permutation Importance")
    st.markdown("*Shows which features are most important for predictions*")
    
    st.markdown("---")
    
    if df_full is None:
        st.warning("⚠️ Dataset not available")
        return
    
    with st.spinner("Calculating permutation importance..."):
        try:
            sample_df = df_full.sample(n=200, random_state=42).copy()
            sample_df = apply_feature_engineering(sample_df)
            
            sample_encoded = pd.get_dummies(sample_df, drop_first=True)
            for col in feature_columns:
                if col not in sample_encoded.columns:
                    sample_encoded[col] = 0
            sample_encoded = sample_encoded[feature_columns]
            X_sample = sample_encoded.values
            y_sample = sample_df['final_exam_score'].values[:len(X_sample)]
            X_sample_scaled = scaler.transform(X_sample)
            
            perm_importance = permutation_importance(
                model, X_sample_scaled, y_sample,
                n_repeats=5, random_state=42, scoring='r2'
            )
            
            perm_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': perm_importance.importances_mean,
                'Std': perm_importance.importances_std
            }).sort_values('Importance', ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(14, 10))
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(perm_df)))[::-1]
            bars = plt.barh(perm_df['Feature'][::-1], perm_df['Importance'][::-1],
                           color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            for bar, val in zip(bars, perm_df['Importance'][::-1]):
                ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Permutation Importance (R² decrease)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Features', fontsize=13, fontweight='bold')
            ax.set_title('Feature Importance - Permutation Importance', fontsize=15, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            add_points(20, "Viewed feature importance")
            st.success("📊 +20 points for exploring feature importance!")
            
            st.markdown("#### 📊 Top 10 Features")
            st.dataframe(perm_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            try:
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(importance_df)))[::-1]
                plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], 
                         color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
                plt.ylabel('Features', fontsize=12, fontweight='bold')
                plt.title('Gradient Boosting Feature Importance', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                add_points(10, "Viewed model importance")
                st.success("📊 +10 points for exploring feature importance!")
            except:
                add_points(5, "Feature importance attempt")
                st.success("🎉 +5 points for effort!")

# ============================================================
# PROGRESS PAGE
# ============================================================
def progress_page():
    st.markdown('<div class="main-header">📈 Progress</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⭐ Total Points", st.session_state.game_score)
    with col2:
        st.metric("🏅 Achievements", len(st.session_state.achievements))
    with col3:
        st.metric("📊 Predictions", st.session_state.predictions_made)
    with col4:
        st.metric("🔥 Best Streak", st.session_state.best_streak)
    
    st.markdown("---")
    
    # Level Progress
    st.markdown("### 📊 Level Progress")
    level_title = get_level_title(st.session_state.level)
    st.markdown(f"**Current Level:** {level_title}")
    progress = (st.session_state.game_score % 200) / 200
    st.progress(progress, text=f"Level {st.session_state.level} → {200 - (st.session_state.game_score % 200)} pts to next level")
    
    # Achievements
    st.markdown("### 🏅 Achievements")
    if st.session_state.achievements:
        cols = st.columns(3)
        for i, ach in enumerate(st.session_state.achievements):
            with cols[i % 3]:
                st.markdown(f"- {ach}")
    else:
        st.info("No achievements yet. Complete challenges to earn them!")
    
    # Activity Log
    st.markdown("### 📜 Activity Log")
    if st.session_state.game_history:
        history_df = pd.DataFrame(st.session_state.game_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No activity yet. Start predicting!")

# ============================================================
# MAIN
# ============================================================
def main():
    choice = show_sidebar()
    
    if choice == "🏠 Home":
        home_page()
    elif choice == "📊 Predict":
        prediction_page()
    elif choice == "🎯 Challenges":
        challenges_page()
    elif choice == "📈 Progress":
        progress_page()

if __name__ == "__main__":
    main()
