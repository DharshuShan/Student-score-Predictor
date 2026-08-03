import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler

# -----------------------------------
# Load dataset
# -----------------------------------

file_path = os.path.join(BASE_DIR, "student_digital_life.csv")

df = pd.read_csv(file_path)

print(f"Dataset shape: {df.shape}")

# -----------------------------------
# Feature Engineering
# -----------------------------------

# Remove identifier columns
drop_cols = ["student_id", "parent_education_level"]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Study efficiency
df["study_efficiency"] = np.clip(
    df["study_hours_per_day"] / (df["sleep_hours"] + 1e-5),
    0,
    10
)

# Focus ratio
df["focus_ratio"] = np.clip(
    df["study_hours_per_day"] / (df["smartphone_usage_hours"] + 1e-5),
    0,
    20
)

# Total screen time
screen_cols = [
    "social_media_hours",
    "gaming_hours",
    "streaming_hours"
]

df["total_screen_time"] = df[screen_cols].sum(axis=1)

# Distraction ratio
df["distraction_ratio"] = (
    df["total_screen_time"]
    / (df["study_hours_per_day"] + 1e-5)
)

# Health score
df["health_score"] = (
    df["sleep_hours"]
    + df["exercise_hours"]
)

# Stress proxy
df["stress_proxy"] = (
    df["caffeine_intake_cups"]
    / (df["sleep_hours"] + 1e-5)
)

# Engagement score
df["engagement_score"] = (
    df[
        [
            "class_attendance_percent",
            "assignment_completion_percent"
        ]
    ].mean(axis=1)
)

# Balance score
df["balance_score"] = (
    df["study_hours_per_day"]
    / (
        df["total_screen_time"]
        + df["sleep_hours"]
        + 1e-5
    )
)

# Interaction features
df["motivation_effect"] = (
    df["motivation_level"]
    * df["study_hours_per_day"]
)

df["study_vs_screen"] = (
    df["study_hours_per_day"]
    / (df["total_screen_time"] + 1e-5)
)

df["sleep_efficiency"] = (
    df["sleep_hours"]
    * df["study_efficiency"]
)

df["mental_pressure"] = (
    df["stress_proxy"]
    * df["distraction_ratio"]
)

df["screen_per_study_hour"] = (
    df["total_screen_time"]
    / (df["study_hours_per_day"] + 1e-5)
)

# -----------------------------------
# One-hot encode categorical features
# -----------------------------------

df_encoded = pd.get_dummies(df, drop_first=True)

# -----------------------------------
# Heatmap of all features
# -----------------------------------

corr_matrix = df_encoded.corr()

plt.figure(figsize=(24, 20))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.1
)

plt.title("Correlation Heatmap - All Features")
plt.tight_layout()
plt.show()

# -----------------------------------
# Feature selection by correlation
# -----------------------------------

target = "final_exam_score"

corr_target = corr_matrix[target].drop(target)

low_corr_features = corr_target[
    (corr_target >= -0.07)
    & (corr_target <= 0.07)
].index.tolist()

print("\nFeatures removed (weak correlation):")
print(low_corr_features)

df_filtered = df_encoded.drop(columns=low_corr_features)

print(f"\nOriginal features: {df_encoded.shape[1] - 1}")
print(f"Removed features: {len(low_corr_features)}")
print(f"Remaining features: {df_filtered.shape[1] - 1}")

# -----------------------------------
# Train-test split
# -----------------------------------

X = df_filtered.drop(columns=target)
y = df_filtered[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------------
# Scale features
# -----------------------------------

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------
# Train Gradient Boosting model
# -----------------------------------

model = GradientBoostingRegressor(
    n_estimators=1300,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.8,
    max_features=0.8,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -----------------------------------
# Save trained model and scaler
# -----------------------------------


model_path = "/Users/dharsh/Documents/Thesis/SHAP Model/final_exam_model.pkl"
scaler_path = "/Users/dharsh/Documents/Thesis/SHAP Model/robust_scaler.pkl"
columns_path = "/Users/dharsh/Documents/Thesis/SHAP Model/model_columns.pkl"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

# Save column order used during training
joblib.dump(X.columns.tolist(), columns_path)

print("\nModel saved successfully")
print(model_path)

# -----------------------------------
# Prediction
# -----------------------------------

y_pred = model.predict(X_test_scaled)

# -----------------------------------
# Evaluation
# -----------------------------------

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Performance")
print("-" * 30)

print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")

# -----------------------------------
# Top feature importances
# -----------------------------------

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Features")
print(importance_df.head(10))

