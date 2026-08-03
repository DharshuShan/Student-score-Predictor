# app_model2.py - Streamlit App for Model 2 with XAI Visualizations
"""
STUDENT SCORE PREDICTOR - Model 2 with XAI Visualizations
Loads Model 2 (final_exam_model1.pkl) with SHAP and 1D PDP
All graphs displayed directly in the frontend

FIX APPLIED (this version): The 1D PDP plots now compute partial
dependence manually via sklearn's `partial_dependence()` function and
convert the resulting grid back to ORIGINAL (unscaled) feature units
using the RobustScaler's center_/scale_ for that single column. This
replaces the previous approach of plotting PartialDependenceDisplay
(which plots in scaled space) and then trying to relabel the ticks
after the fact -- that produced a mismatch between tick positions and
the actual scaled grid, which is why the x-axis looked compressed /
wrong. The new approach builds the correct axis directly, so no
post-hoc tick patching is needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shap
from sklearn.inspection import partial_dependence
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🎓 Student Score Predictor - Model 2",
    page_icon="📊",
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
        .stButton>button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .tab-content {
            padding: 1rem 0;
        }
        .viz-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .stForm {
            border: none !important;
            padding: 0 !important;
        }
        .pdp-plot-container {
            margin-top: 1rem;
            padding: 1rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
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
    """Load Model 2, scaler, and feature columns"""

    model_path = '/Users/dharsh/Documents/Thesis/SHAP Model/final_exam_model1.pkl'
    scaler_path = '/Users/dharsh/Documents/Thesis/SHAP Model/robust_scaler1.pkl'
    columns_path = '/Users/dharsh/Documents/Thesis/SHAP Model/model_columns1.pkl'

    print(f"📂 Looking for model at: {model_path}")

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        st.info("Please run the Model 2 training script first.")
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
# LOAD DATASET FOR PDP
# ============================================================
@st.cache_data
def load_dataset():
    """Load the student dataset for PDP"""
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
df_full = load_dataset()

# Show status in sidebar
if model is not None:
    st.sidebar.success("✅ Model 2 loaded")
else:
    st.sidebar.error("❌ Model not loaded")

if df_full is not None:
    st.sidebar.success(f"✅ Dataset: {len(df_full)} students")
else:
    st.sidebar.warning("⚠️ Dataset not loaded")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def format_hours(hours):
    """Format hours as hours and minutes for display"""
    hrs = int(hours)
    mins = int(round((hours - hrs) * 60))
    if hrs > 0 and mins > 0:
        return f"{hrs}h {mins}m"
    elif hrs > 0:
        return f"{hrs}h"
    elif mins > 0:
        return f"{mins}m"
    else:
        return "0m"

def time_input_combined(label, max_hours, default_hours, key_prefix):
    """Create time input with hours and minutes"""
    default_hrs = int(default_hours)
    default_mins = int((default_hours - default_hrs) * 60)
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
        mins = st.selectbox(
            f"{label} (Minutes)",
            options=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            index=default_mins // 5,
            key=f"{key_prefix}_mins"
        )

    with col3:
        total_hours = hrs + mins / 60
        st.markdown(f"<div style='margin-top: 25px; font-weight: bold;'>⏱️ {format_hours(total_hours)}</div>",
                   unsafe_allow_html=True)

    return total_hours

def engineer_features(input_dict):
    """Apply feature engineering (matching Model 2 training)"""
    df = pd.DataFrame([input_dict])

    df["study_efficiency"] = np.clip(
        df["study_hours_per_day"] / (df["sleep_hours"] + 1e-5), 0, 10
    )
    df["focus_ratio"] = np.clip(
        df["study_hours_per_day"] / (df["smartphone_usage_hours"] + 1e-5), 0, 20
    )
    df["total_screen_time"] = (
        df["social_media_hours"] + df["gaming_hours"] + df["streaming_hours"]
    )
    df["distraction_ratio"] = df["total_screen_time"] / (df["study_hours_per_day"] + 1e-5)
    df["health_score"] = df["sleep_hours"] + df["exercise_hours"]
    df["stress_proxy"] = df["caffeine_intake_cups"] / (df["sleep_hours"] + 1e-5)
    df["engagement_score"] = (
        df["class_attendance_percent"] + df["assignment_completion_percent"]
    ) / 2
    df["balance_score"] = df["study_hours_per_day"] / (
        df["total_screen_time"] + df["sleep_hours"] + 1e-5
    )
    df["motivation_effect"] = df["motivation_level"] * df["study_hours_per_day"]
    df["study_vs_screen"] = df["study_hours_per_day"] / (df["total_screen_time"] + 1e-5)
    df["sleep_efficiency"] = df["sleep_hours"] * df["study_efficiency"]
    df["mental_pressure"] = df["stress_proxy"] * df["distraction_ratio"]
    df["screen_per_study_hour"] = df["total_screen_time"] / (df["study_hours_per_day"] + 1e-5)

    # One-hot encode
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Align with model columns
    if feature_columns is not None:
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_columns]

    return df_encoded

def engineer_features_bulk(df):
    """Apply feature engineering to entire dataset"""
    df_eng = df.copy()

    # Apply feature engineering
    df_eng["study_efficiency"] = np.clip(
        df_eng["study_hours_per_day"] / (df_eng["sleep_hours"] + 1e-5), 0, 10
    )
    df_eng["focus_ratio"] = np.clip(
        df_eng["study_hours_per_day"] / (df_eng["smartphone_usage_hours"] + 1e-5), 0, 20
    )
    df_eng["total_screen_time"] = (
        df_eng["social_media_hours"] + df_eng["gaming_hours"] + df_eng["streaming_hours"]
    )
    df_eng["distraction_ratio"] = df_eng["total_screen_time"] / (df_eng["study_hours_per_day"] + 1e-5)
    df_eng["health_score"] = df_eng["sleep_hours"] + df_eng["exercise_hours"]
    df_eng["stress_proxy"] = df_eng["caffeine_intake_cups"] / (df_eng["sleep_hours"] + 1e-5)
    df_eng["engagement_score"] = (
        df_eng["class_attendance_percent"] + df_eng["assignment_completion_percent"]
    ) / 2
    df_eng["balance_score"] = df_eng["study_hours_per_day"] / (
        df_eng["total_screen_time"] + df_eng["sleep_hours"] + 1e-5
    )
    df_eng["motivation_effect"] = df_eng["motivation_level"] * df_eng["study_hours_per_day"]
    df_eng["study_vs_screen"] = df_eng["study_hours_per_day"] / (df_eng["total_screen_time"] + 1e-5)
    df_eng["sleep_efficiency"] = df_eng["sleep_hours"] * df_eng["study_efficiency"]
    df_eng["mental_pressure"] = df_eng["stress_proxy"] * df_eng["distraction_ratio"]
    df_eng["screen_per_study_hour"] = df_eng["total_screen_time"] / (df_eng["study_hours_per_day"] + 1e-5)

    # One-hot encode
    df_encoded = pd.get_dummies(df_eng, drop_first=True)

    # Ensure all required columns exist
    if feature_columns is not None:
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

    # Select only the columns needed by the model
    df_encoded = df_encoded[feature_columns]

    return df_encoded

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_score(student_data):
    """Make prediction using Model 2"""
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
# SHAP VISUALIZATIONS (Only 2 plots)
# ============================================================
@st.cache_resource
def get_shap_explainer():
    """Create and cache SHAP explainer"""
    if model is None:
        return None
    return shap.TreeExplainer(model)

def generate_shap_plots(student_data, score):
    """Generate SHAP Bar and Waterfall plots"""

    if model is None or scaler is None:
        st.warning("⚠️ Model not available for SHAP")
        return

    try:
        # Prepare data
        features = engineer_features(student_data)
        features_scaled = scaler.transform(features)
        X_input = pd.DataFrame(features_scaled, columns=feature_columns)

        # Get SHAP explainer
        explainer = get_shap_explainer()
        if explainer is None:
            st.warning("⚠️ SHAP explainer not available")
            return

        shap_values = explainer.shap_values(X_input)

        # Get expected value
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]
        elif hasattr(expected_value, '__len__') and len(expected_value) > 0:
            expected_value = expected_value[0]

        # 1. SHAP Summary Bar Plot
        st.markdown("### 📊 SHAP Feature Importance Ranking")
        st.markdown("*Ranked importance of features based on SHAP values*")

        fig, ax = plt.subplots(figsize=(12, 10))
        shap.summary_plot(shap_values, X_input, feature_names=feature_columns,
                          show=False, max_display=15, plot_type="bar")
        plt.title(f'SHAP Feature Importance Ranking\nPredicted Score: {score:.2f}/100',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # 2. SHAP Waterfall Plot
        st.markdown("### 📊 SHAP Waterfall Plot")
        st.markdown("*Prediction breakdown showing how each feature contributed to this specific prediction*")

        if isinstance(shap_values, list):
            shap_values_for_plot = shap_values[0]
        else:
            shap_values_for_plot = shap_values

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_for_plot[0] if len(shap_values_for_plot.shape) > 1 else shap_values_for_plot,
                base_values=expected_value,
                data=X_input.iloc[0],
                feature_names=feature_columns
            ),
            show=False,
            max_display=15
        )
        plt.title(f'SHAP Waterfall Plot - Prediction Breakdown\nPredicted Score: {score:.2f}/100',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"❌ Error generating SHAP plots: {e}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================
# 1D PDP VISUALIZATION (original feature units on the x-axis)
# ============================================================
def _compute_1d_pdp(feature, X_scaled_df):
    """
    Compute a single-feature partial dependence curve and return it with
    the grid expressed in ORIGINAL (unscaled) feature units.

    Returns: (grid_original, pdp_values)
    """
    col_idx = feature_columns.index(feature)

    pd_result = partial_dependence(
        model,
        X_scaled_df,
        features=[feature],
        kind='average',
        grid_resolution=30
    )

    # sklearn >=1.3 uses 'grid_values'; older versions use 'values'
    grid_scaled = pd_result.get('grid_values', pd_result.get('values'))[0]
    pdp_values = pd_result['average'][0]

    # Convert grid back to original units using the scaler's per-feature
    # center/scale (RobustScaler: x_original = x_scaled * scale_ + center_)
    if hasattr(scaler, 'center_') and hasattr(scaler, 'scale_'):
        center = scaler.center_[col_idx]
        spread = scaler.scale_[col_idx]
        grid_original = grid_scaled * spread + center
    else:
        # Fallback: leave as-is if scaler doesn't expose center_/scale_
        grid_original = grid_scaled

    return grid_original, pdp_values


def generate_1d_pdp_plots(feature1, feature2):
    """Generate two 1D PDP plots side by side, with the x-axis expressed
    in the ORIGINAL (unscaled) feature units rather than the internal
    scaled representation used by the model."""

    if model is None or scaler is None or df_full is None:
        st.warning("⚠️ Model or dataset not available for PDP visualization")
        return

    # Check if features exist in model columns
    if feature1 not in feature_columns:
        st.error(f"❌ Feature '{feature1}' not found in model")
        return
    if feature2 not in feature_columns:
        st.error(f"❌ Feature '{feature2}' not found in model")
        return

    try:
        # Sample data for faster computation
        sample_df = df_full.sample(n=500, random_state=42).copy()

        # Apply feature engineering to get all model features
        df_engineered = engineer_features_bulk(sample_df)

        # Scale the features (this is what the model was actually trained on)
        X_scaled = scaler.transform(df_engineered)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)

        # Compute PDP curves for both features, already converted to
        # original units
        grid1, pdp1 = _compute_1d_pdp(feature1, X_scaled_df)
        grid2, pdp2 = _compute_1d_pdp(feature2, X_scaled_df)

        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # --- Plot 1 ---
        ax1.plot(grid1, pdp1, color='#1f77b4', linewidth=2.2)
        ax1.set_title(f'PDP: {feature1.replace("_", " ").title()}',
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel(feature1.replace('_', ' ').title(), fontsize=10)
        ax1.set_ylabel('Average Predicted Score', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # --- Plot 2 ---
        ax2.plot(grid2, pdp2, color='#2ca02c', linewidth=2.2)
        ax2.set_title(f'PDP: {feature2.replace("_", " ").title()}',
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel(feature2.replace('_', ' ').title(), fontsize=10)
        ax2.set_ylabel('Average Predicted Score', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.suptitle('1D Partial Dependence Plots (PDP) with Original Feature Values',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Display the plots
        st.pyplot(fig)
        plt.close()

        # Show feature value ranges (from the raw dataset, for reference)
        if feature1 in sample_df.columns and feature2 in sample_df.columns:
            st.info(f"""
            **📊 Feature Value Ranges (from dataset):**
            - **{feature1.replace('_', ' ').title()}**: {sample_df[feature1].min():.1f} to {sample_df[feature1].max():.1f}
              (Mean: {sample_df[feature1].mean():.1f})
            - **{feature2.replace('_', ' ').title()}**: {sample_df[feature2].min():.1f} to {sample_df[feature2].max():.1f}
              (Mean: {sample_df[feature2].mean():.1f})
            """)
        else:
            st.info(
                "ℹ️ One or both selected features are engineered (derived) "
                "features, so their axis reflects the engineered feature's "
                "own scale rather than a raw input column."
            )

        st.success(f"✅ 1D PDP plots generated for: **{feature1.replace('_', ' ').title()}** and **{feature2.replace('_', ' ').title()}**")
        st.caption("💡 Each plot shows how the predicted score changes as a single feature varies, holding all other features at their observed values. The x-axis is shown in the feature's original (unscaled) units.")

    except Exception as e:
        st.error(f"❌ Error generating 1D PDP plots: {e}")
        import traceback
        st.code(traceback.format_exc())

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
    if 'pdp_features' not in st.session_state:
        st.session_state.pdp_features = (None, None)
    if 'show_pdp_plot' not in st.session_state:
        st.session_state.show_pdp_plot = False
    if 'current_score' not in st.session_state:
        st.session_state.current_score = None
    if 'current_student_data' not in st.session_state:
        st.session_state.current_student_data = None

init_session_state()

# ============================================================
# SIDEBAR
# ============================================================
def show_sidebar():
    with st.sidebar:
        st.markdown("### 🎮 XAI Prediction Dashboard")
        st.markdown(f"**📊 Predictions:** {st.session_state.predictions_made}")

        st.markdown("---")
        menu = ["🏠 Home", "📊 Predict"]
        choice = st.radio("Navigate", menu, index=0)

        if st.button("🔄 Reset Prediction"):
            st.session_state.predictions_made = 0
            st.session_state.history = []
            st.session_state.current_score = None
            st.session_state.current_student_data = None
            st.session_state.show_pdp_plot = False
            st.session_state.pdp_features = (None, None)
            st.rerun()

        return choice

# ============================================================
# HOME PAGE
# ============================================================
def home_page():
    st.markdown('<div class="main-header">🎓 Student Score Predictor - Model 2</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3>📊 Make Predictions</h3>
            <p>Enter student data and get AI-powered exam score predictions with full XAI visualizations.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h3>🔮 Understand Why</h3>
            <p>Get SHAP and 1D PDP visualizations to understand the prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 XAI Visualizations Included")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**SHAP Visualizations**")
        st.markdown("1. Feature Importance Bar Plot")
        st.markdown("2. Waterfall Plot")

    with col2:
        st.markdown("**PDP Visualizations**")
        st.markdown("3. 1D Partial Dependence Plots (on-demand)")

# ============================================================
# RESULTS SECTION
# ============================================================
def render_results_section():
    score = st.session_state.current_score
    student_data = st.session_state.current_student_data

    if score is None or student_data is None:
        return

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
            <div class="score-number">{score:.1f}</div>
            <div class="grade-badge">{grade}</div>
        </div>
        """, unsafe_allow_html=True)

    # Input Summary
    st.markdown("---")
    st.markdown("### 📊 Input Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Study Hours", f"{student_data['study_hours_per_day']:.1f}h")
        st.metric("Sleep Hours", f"{student_data['sleep_hours']:.1f}h")
    with col2:
        st.metric("Phone Usage", f"{student_data['smartphone_usage_hours']:.1f}h")
        st.metric("Social Media", f"{student_data['social_media_hours']:.1f}h")
    with col3:
        st.metric("Attendance", f"{student_data['class_attendance_percent']}%")
        st.metric("Assignments", f"{student_data['assignment_completion_percent']}%")
    with col4:
        st.metric("Motivation", f"{student_data['motivation_level']}/10")
        st.metric("Gaming", f"{student_data['gaming_hours']:.1f}h")

    # ============================================================
    # XAI VISUALIZATIONS SECTION
    # ============================================================
    st.markdown("---")
    st.markdown('<div style="text-align: center; font-size: 2rem; font-weight: bold; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">🔍 XAI Visualizations</div>', unsafe_allow_html=True)

    # SHAP Visualizations
    with st.expander("📊 SHAP Visualizations (Click to expand)", expanded=True):
        st.markdown("#### SHAP (SHapley Additive exPlanations)")
        st.markdown("*SHAP explains predictions by showing how each feature contributes to the final score*")
        generate_shap_plots(student_data, score)

    # ============================================================
    # 1D PDP Visualizations
    # ============================================================
    with st.expander("🟢 1D PDP Visualization (Click to expand)", expanded=False):
        st.markdown("#### 1D PDP (Partial Dependence Plot)")
        st.markdown("*1D PDP shows how the predicted score changes when a single feature varies*")

        # Get available features - filter out one-hot encoded features
        available_features = [f for f in feature_columns if not f.startswith('gender_') and not f.startswith('mental_') and not f.startswith('internet_')]

        if not available_features:
            available_features = feature_columns[:20]

        if len(available_features) >= 2:
            st.markdown("##### Select two features for 1D PDP analysis")
            st.caption("Two separate 1D PDP plots will be generated side by side, with the x-axis shown in original feature units")

            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox(
                    "Select First Feature",
                    available_features,
                    index=0,
                    key="pdp_feature1_final"
                )

            with col2:
                feature2 = st.selectbox(
                    "Select Second Feature",
                    available_features,
                    index=1 if len(available_features) > 1 else 0,
                    key="pdp_feature2_final"
                )

            if feature1 == feature2:
                st.warning("⚠️ Please select two different features")
                st.button("📊 Generate 1D PDP Plots", key="pdp_generate_disabled", disabled=True)
            else:
                if st.button("📊 Generate 1D PDP Plots", key="pdp_generate_final"):
                    st.session_state.pdp_features = (feature1, feature2)
                    st.session_state.show_pdp_plot = True

            if st.session_state.show_pdp_plot:
                f1, f2 = st.session_state.pdp_features
                if f1 == feature1 and f2 == feature2:
                    st.markdown("---")
                    st.markdown('<div class="pdp-plot-container">', unsafe_allow_html=True)
                    st.markdown("##### 📊 1D PDP Plots")
                    with st.spinner("Generating 1D PDP plots..."):
                        generate_1d_pdp_plots(f1, f2)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.session_state.show_pdp_plot = False
                    st.session_state.pdp_features = (None, None)
        else:
            st.warning("⚠️ Not enough features available for PDP visualization")

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
        st.info("ℹ️ **Note:** Total hours cannot exceed 24 hours per day. Gaming and Streaming can overlap. Time inputs use 5-minute increments.")

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

        # Validation
        max_entertainment = max(gaming, streaming)
        total_hours = study_hours + smartphone_usage + social_media + sleep + exercise + max_entertainment

        col1, col2 = st.columns(2)

        with col1:
            if total_hours > 24:
                st.error(f"⚠️ **Total exceeds 24h!** Current: {format_hours(total_hours)}")
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
            st.caption(f"🎮 Entertainment: {format_hours(max_entertainment)}")
            st.markdown("</div>", unsafe_allow_html=True)

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

        if total_hours > 24:
            st.warning("⚠️ Please adjust hours to be within 24 hours.")
            st.button("🔮 Predict Score!", use_container_width=True, disabled=True)
        else:
            if st.button("🔮 Predict Score!", use_container_width=True):
                with st.spinner("Analyzing student data..."):
                    score = predict_score(student_data)

                    if score is not None:
                        st.session_state.predictions_made += 1
                        st.session_state.current_score = score
                        st.session_state.current_student_data = student_data
                        st.session_state.show_pdp_plot = False
                        st.session_state.pdp_features = (None, None)
                    else:
                        st.error("❌ Prediction failed. Please try again.")

    render_results_section()

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
