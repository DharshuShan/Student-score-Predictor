import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

# Try importing LIME, but if it fails, proceed without it
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except:
    LIME_AVAILABLE = False
    print("LIME not available - will skip LIME analysis")

# -----------------------------------
# Load dataset
# -----------------------------------

file_path = "/Users/dharsh/Documents/Thesis/SHAP Model/student_digital_life.csv"
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
    0, 10
)

# Focus ratio
df["focus_ratio"] = np.clip(
    df["study_hours_per_day"] / (df["smartphone_usage_hours"] + 1e-5),
    0, 20
)

# Total screen time
screen_cols = ["social_media_hours", "gaming_hours", "streaming_hours"]
df["total_screen_time"] = df[screen_cols].sum(axis=1)

# Distraction ratio
df["distraction_ratio"] = df["total_screen_time"] / (df["study_hours_per_day"] + 1e-5)

# Health score
df["health_score"] = df["sleep_hours"] + df["exercise_hours"]

# Stress proxy
df["stress_proxy"] = df["caffeine_intake_cups"] / (df["sleep_hours"] + 1e-5)

# Engagement score
df["engagement_score"] = df[["class_attendance_percent", "assignment_completion_percent"]].mean(axis=1)

# Balance score
df["balance_score"] = df["study_hours_per_day"] / (df["total_screen_time"] + df["sleep_hours"] + 1e-5)

# Interaction features
df["motivation_effect"] = df["motivation_level"] * df["study_hours_per_day"]
df["study_vs_screen"] = df["study_hours_per_day"] / (df["total_screen_time"] + 1e-5)
df["sleep_efficiency"] = df["sleep_hours"] * df["study_efficiency"]
df["mental_pressure"] = df["stress_proxy"] * df["distraction_ratio"]
df["screen_per_study_hour"] = df["total_screen_time"] / (df["study_hours_per_day"] + 1e-5)

# -----------------------------------
# One-hot encode categorical features
# -----------------------------------

df_encoded = pd.get_dummies(df, drop_first=True)

# -----------------------------------
# Feature selection by correlation
# -----------------------------------

target = "final_exam_score"
corr_matrix = df_encoded.corr()
corr_target = corr_matrix[target].drop(target)

low_corr_features = corr_target[(corr_target >= -0.07) & (corr_target <= 0.07)].index.tolist()
print(f"\nFeatures removed (weak correlation): {len(low_corr_features)}")

df_filtered = df_encoded.drop(columns=low_corr_features)
print(f"Remaining features: {df_filtered.shape[1] - 1}")

# -----------------------------------
# Train-test split
# -----------------------------------

X = df_filtered.drop(columns=target)
y = df_filtered[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -----------------------------------
# Scale features
# -----------------------------------

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for XAI
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

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

model_path = "/Users/dharsh/Documents/Thesis/SHAP Model/final_exam_model1.pkl"
scaler_path = "/Users/dharsh/Documents/Thesis/SHAP Model/robust_scaler1.pkl"
columns_path = "/Users/dharsh/Documents/Thesis/SHAP Model/model_columns1.pkl"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(X.columns.tolist(), columns_path)
print("\nModel saved successfully")

# -----------------------------------
# Model Performance
# -----------------------------------

y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"R² Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MAE      : {mae:.4f}")
print("="*50)

# ==========================================
# FEATURE IMPORTANCE FROM MODEL
# ==========================================

print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\nTop 10 Features (Gradient Boosting)")
print(importance_df.head(10))

# ==========================================
# XAI TECHNIQUES (No SHAP required)
# ==========================================

print("\n" + "="*50)
print("XAI ANALYSIS")
print("="*50)

# -----------------------------------
# 1. Permutation Importance
# -----------------------------------

print("\n1. Permutation Importance Analysis...")

perm_importance = permutation_importance(
    model, 
    X_test_scaled, 
    y_test, 
    n_repeats=30, 
    random_state=42,
    scoring='r2'
)

# Create permutation importance DataFrame
perm_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

# Plot permutation importance
plt.figure(figsize=(12, 8))
top_n = min(15, len(perm_df))
plt.barh(perm_df['Feature'][:top_n][::-1], perm_df['Importance'][:top_n][::-1],
         xerr=perm_df['Std'][:top_n][::-1])
plt.xlabel('Permutation Importance (R² decrease)')
plt.title('Permutation Feature Importance - Top 15 Features')
plt.tight_layout()
plt.savefig("/Users/dharsh/Documents/Thesis/SHAP Model/permutation_importance.png", dpi=300, bbox_inches='tight')
plt.close()

perm_df.to_csv("/Users/dharsh/Documents/Thesis/SHAP Model/permutation_importance.csv", index=False)
print("✓ Permutation importance saved")

# -----------------------------------
# 2. LIME Analysis (if available)
# -----------------------------------

if LIME_AVAILABLE:
    print("\n2. LIME Analysis...")
    
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_scaled,
        feature_names=X.columns.tolist(),
        mode='regression',
        discretize_continuous=True,
        kernel_width=0.5
    )
    
    # Select a few test instances for explanation
    lime_samples = [0, 10, 20, 50, 100]
    
    for idx in lime_samples:
        if idx < len(X_test_df):
            exp = lime_explainer.explain_instance(
                X_test_scaled[idx],
                model.predict,
                num_features=10
            )
            
            exp.save_to_file(f"/Users/dharsh/Documents/Thesis/SHAP Model/lime_explanation_{idx}.html")
            
            fig = exp.as_pyplot_figure()
            plt.title(f"LIME Explanation - Sample {idx}")
            plt.tight_layout()
            plt.savefig(f"/Users/dharsh/Documents/Thesis/SHAP Model/lime_plot_{idx}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print("✓ LIME explanations saved")
    
    # LIME feature importance aggregation
    feature_weights = {feat: 0 for feat in X.columns}
    for idx in range(min(50, len(X_test_df))):
        exp = lime_explainer.explain_instance(X_test_scaled[idx], model.predict, num_features=10)
        for feature, weight in exp.as_list():
            if feature in feature_weights:
                feature_weights[feature] += abs(weight)
    
    lime_importance = pd.DataFrame({
        'Feature': list(feature_weights.keys()),
        'LIME_Importance': list(feature_weights.values())
    }).sort_values('LIME_Importance', ascending=False)
    
    lime_importance.to_csv("/Users/dharsh/Documents/Thesis/SHAP Model/lime_feature_importance.csv", index=False)
    print("✓ LIME feature importance saved")
else:
    print("\n2. LIME Analysis: Skipped (not available)")

# -----------------------------------
# 3. Partial Dependence Plots (PDP)
# -----------------------------------

print("\n3. Partial Dependence Plots...")

# Select top 6 features for PDP
top_feature_names = importance_df.head(6)['Feature'].tolist()

for feature in top_feature_names:
    plt.figure(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(
        model, 
        X_test_scaled, 
        features=[feature],
        feature_names=X.columns,
        kind='average'
    )
    plt.title(f"Partial Dependence Plot - {feature}")
    plt.tight_layout()
    plt.savefig(f"/Users/dharsh/Documents/Thesis/SHAP Model/pdp_{feature}.png", dpi=300, bbox_inches='tight')
    plt.close()

# 2-Way PDP for top interactions
if len(top_feature_names) >= 2:
    plt.figure(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(
        model,
        X_test_scaled,
        features=[(top_feature_names[0], top_feature_names[1])],
        feature_names=X.columns,
        kind='average'
    )
    plt.title(f"2-Way PDP - {top_feature_names[0]} vs {top_feature_names[1]}")
    plt.tight_layout()
    plt.savefig("/Users/dharsh/Documents/Thesis/SHAP Model/pdp_interaction.png", dpi=300, bbox_inches='tight')
    plt.close()

print("✓ PDP plots saved")

# -----------------------------------
# 4. Feature Importance Comparison
# -----------------------------------

print("\n4. Comparing Feature Importance Methods...")

# Create comparison DataFrame
feature_importance_comp = pd.DataFrame({
    'Feature': X.columns,
    'Gradient_Boosting': model.feature_importances_,
    'Permutation': perm_df.set_index('Feature')['Importance'].reindex(X.columns).values
})

# Sort and save
feature_importance_comp = feature_importance_comp.sort_values('Gradient_Boosting', ascending=False)
feature_importance_comp.to_csv("/Users/dharsh/Documents/Thesis/SHAP Model/feature_importance_comparison.csv", index=False)

# Plot comparison
plt.figure(figsize=(14, 10))
top_10_features = feature_importance_comp.head(10)['Feature'].tolist()
comparison_data = feature_importance_comp[feature_importance_comp['Feature'].isin(top_10_features)]

# Normalize for visualization
for col in ['Gradient_Boosting', 'Permutation']:
    comparison_data[f'{col}_norm'] = comparison_data[col] / comparison_data[col].max()

comparison_data_melted = pd.melt(
    comparison_data,
    id_vars=['Feature'],
    value_vars=['Gradient_Boosting_norm', 'Permutation_norm'],
    var_name='Method',
    value_name='Normalized_Importance'
)

plt.figure(figsize=(12, 8))
sns.barplot(data=comparison_data_melted, x='Feature', y='Normalized_Importance', hue='Method')
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance Comparison - Top 10 Features')
plt.legend()
plt.tight_layout()
plt.savefig("/Users/dharsh/Documents/Thesis/SHAP Model/importance_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Importance comparison saved")

# -----------------------------------
# 5. Additional: Feature Importance Visualization
# -----------------------------------

print("\n5. Generating Feature Importance Visualization...")

# Plot top 15 features from Gradient Boosting
plt.figure(figsize=(12, 8))
top_15 = importance_df.head(15)
plt.barh(top_15['Feature'], top_15['Importance'])
plt.xlabel('Feature Importance')
plt.title('Gradient Boosting Feature Importance - Top 15 Features')
plt.tight_layout()
plt.savefig("/Users/dharsh/Documents/Thesis/SHAP Model/gb_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Feature importance visualization saved")

# -----------------------------------
# Save all results
# -----------------------------------

print("\n" + "="*50)
print("ALL XAI ANALYSIS COMPLETED!")
print("="*50)
print("\nSaved files:")
print("1. Model Performance Metrics (above)")
print("2. Gradient Boosting Feature Importance (top 10 displayed)")
print("3. Permutation Importance: plot and CSV")
if LIME_AVAILABLE:
    print("4. LIME explanations: HTML and PNG files")
    print("5. LIME feature importance: CSV")
print("6. Partial Dependence Plots: PNG files")
print("7. Feature Importance Comparison: CSV and plot")
print("8. Gradient Boosting Feature Importance Plot: PNG")
print("\nAll files saved to: /Users/dharsh/Documents/Thesis/SHAP Model/")
print("="*50)