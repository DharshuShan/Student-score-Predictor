import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# XAI Libraries
import shap
from lime.lime_tabular import LimeTabularExplainer
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# 📊 Load and prepare data (reusing your feature engineering)
# ---------------------------

df = pd.read_csv('/Users/dharsh/Documents/Thesis/XAI Model/student_digital_life.csv')

# Feature engineering (same as your original)
df = df.drop(columns=['student_id', 'age', 'parent_education_level'])

df['study_efficiency'] = df['study_hours_per_day'] / (df['sleep_hours'] + 1)
df['focus_ratio'] = df['study_hours_per_day'] / (df['smartphone_usage_hours'] + 1)
df['total_screen_time'] = df['social_media_hours'] + df['gaming_hours'] + df['streaming_hours']
df['distraction_ratio'] = df['total_screen_time'] / (df['study_hours_per_day'] + 1)
df['health_score'] = df['sleep_hours'] + df['exercise_hours']
df['stress_proxy'] = df['caffeine_intake_cups'] / (df['sleep_hours'] + 1)
df['engagement_score'] = (df['class_attendance_percent'] + df['assignment_completion_percent']) / 2
df['balance_score'] = df['study_hours_per_day'] / (df['total_screen_time'] + df['sleep_hours'] + 1)
df['motivation_effect'] = df['motivation_level'] * df['study_hours_per_day']
df['study_vs_screen'] = df['study_hours_per_day'] / (df['total_screen_time'] + 1)
df['sleep_efficiency'] = df['sleep_hours'] * df['study_efficiency']
df['mental_pressure'] = df['stress_proxy'] * df['distraction_ratio']

df = df.drop(columns=[
    'class_attendance_percent', 'gaming_hours', 'streaming_hours',
    'smartphone_usage_hours', 'social_media_hours'
])

# Prepare data
X = df.drop('final_exam_score', axis=1)
y = df['final_exam_score']
X = pd.get_dummies(X, drop_first=True)

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'model_columns.pkl')

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (your original)
model = GradientBoostingRegressor(
    n_estimators=1500, learning_rate=0.02, max_depth=3,
    subsample=0.8, max_features=0.8, random_state=42
)
model.fit(X_train, y_train)

# ---------------------------
# 🔮 XAI: SHAP Analysis
# ---------------------------

print("\n" + "="*60)
print("🔮 SHAP ANALYSIS (Global & Local Explanations)")
print("="*60)

# Create SHAP explainer
explainer_shap = shap.TreeExplainer(model)
shap_values = explainer_shap.shap_values(X_test)

# 1. Global Feature Importance
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.title("SHAP Feature Importance (Global)", fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Bar plot of mean absolute SHAP values
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
plt.title("Mean |SHAP| Feature Importance", fontsize=14)
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300)
plt.show()

# 3. Feature importance ranking
mean_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Value': mean_shap
}).sort_values('SHAP_Value', ascending=False)

print("\n📊 Top 10 Most Important Features (SHAP):")
print(shap_importance.head(10).to_string(index=False))

# 4. Individual predictions explanation (Local)
sample_idx = 0  # Explain first test sample
sample_data = X_test.iloc[sample_idx:sample_idx+1]
sample_pred = model.predict(sample_data)[0]

plt.figure(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(values=shap_values[sample_idx], 
                     base_values=explainer_shap.expected_value,
                     data=sample_data.values[0],
                     feature_names=feature_names),
    show=False
)
plt.title(f"SHAP Waterfall Plot (Prediction: {sample_pred:.2f})", fontsize=12)
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=300)
plt.show()

# ---------------------------
# 📋 LIME Analysis (FIXED)
# ---------------------------

print("\n" + "="*60)
print("📋 LIME ANALYSIS (Local Interpretable Explanations)")
print("="*60)

# Create LIME explainer
lime_explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    mode='regression',
    training_labels=y_train.values,
    random_state=42
)

# Explain multiple test samples
num_explanations = 3
for i in range(min(num_explanations, len(X_test))):
    exp = lime_explainer.explain_instance(
        X_test.iloc[i].values,
        model.predict,
        num_features=10
    )
    
    print(f"\n🔍 LIME Explanation for Sample {i+1}:")
    print(f"   Actual Score: {y_test.iloc[i]:.2f}")
    print(f"   Predicted Score: {model.predict(X_test.iloc[i:i+1])[0]:.2f}")
    
    # FIXED: Use show_in_notebook or save_to_file instead of as_matplotlib
    # Method 1: Save as HTML (interactive)
    exp.save_to_file(f'lime_explanation_sample_{i+1}.html')
    
    # Method 2: Create matplotlib figure manually
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract feature contributions
    contributions = exp.as_list()
    features = [c[0] for c in contributions[:10]]
    values = [c[1] for c in contributions[:10]]
    
    # Create horizontal bar chart
    colors = ['green' if v > 0 else 'red' for v in values]
    ax.barh(features, values, color=colors)
    ax.set_xlabel('Feature Contribution')
    ax.set_title(f'LIME Feature Contributions - Sample {i+1}\nPrediction: {model.predict(X_test.iloc[i:i+1])[0]:.2f}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'lime_explanation_sample_{i+1}.png', dpi=300)
    plt.show()
    
    # Method 3: Print contributions (optional)
    print(f"\n   Top contributing features:")
    for feature, value in contributions[:5]:
        direction = "↑ increases" if value > 0 else "↓ decreases"
        print(f"   • {feature}: {value:.3f} ({direction} prediction)")

# ---------------------------
# 🎯 Partial Dependence Plots (PDP)
# ---------------------------

print("\n" + "="*60)
print("🎯 PARTIAL DEPENDENCE PLOTS")
print("="*60)

from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# Select top 6 features for PDP
top_features = shap_importance.head(6)['Feature'].tolist()
feature_indices = [feature_names.index(f) for f in top_features if f in feature_names]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, (feat_name, feat_idx) in enumerate(zip(top_features[:6], feature_indices[:6])):
    PartialDependenceDisplay.from_estimator(
        model, X_train, [feat_idx],
        ax=axes[idx],
        kind="average",
        grid_resolution=20
    )
    axes[idx].set_title(f"PDP: {feat_name}", fontsize=12)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle("Partial Dependence Plots - Top Features", fontsize=16)
plt.tight_layout()
plt.savefig('pdp_plots.png', dpi=300)
plt.show()

# ---------------------------
# 🔥 Feature Interaction Analysis
# ---------------------------

print("\n" + "="*60)
print("🔥 FEATURE INTERACTION ANALYSIS")
print("="*60)

# SHAP interaction values
try:
    shap_interaction = explainer_shap.shap_interaction_values(X_test)
    
    # Get top 2 features for interaction
    top2 = shap_importance.head(2)['Feature'].tolist()
    idx1, idx2 = feature_names.index(top2[0]), feature_names.index(top2[1])
    
    # Create interaction plot
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        idx1, shap_values, X_test,
        interaction_index=idx2,
        feature_names=feature_names,
        show=False
    )
    plt.title(f"SHAP Interaction: {top2[0]} ↔ {top2[1]}", fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_interaction.png', dpi=300)
    plt.show()
except Exception as e:
    print(f"Note: SHAP interaction plot requires more memory. Skipping: {e}")

# ---------------------------
# 📊 Model Performance & Confidence
# ---------------------------

print("\n" + "="*60)
print("📊 MODEL PERFORMANCE & UNCERTAINTY")
print("="*60)

# Prediction intervals using quantile regression (simplified)
from sklearn.ensemble import GradientBoostingRegressor

# Train multiple models for uncertainty estimation
predictions = []
for seed in [42, 43, 44, 45, 46]:
    model_ensemble = GradientBoostingRegressor(
        n_estimators=1500, learning_rate=0.02, max_depth=3,
        subsample=0.8, max_features=0.8, random_state=seed
    )
    model_ensemble.fit(X_train, y_train)
    predictions.append(model_ensemble.predict(X_test))

predictions = np.array(predictions)
pred_mean = predictions.mean(axis=0)
pred_std = predictions.std(axis=0)
pred_lower = pred_mean - 1.96 * pred_std
pred_upper = pred_mean + 1.96 * pred_std

# Plot prediction intervals
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(y_test.values)

plt.errorbar(
    range(len(y_test)), 
    pred_mean[sorted_idx],
    yerr=1.96 * pred_std[sorted_idx],
    fmt='o', alpha=0.5, capsize=2, markersize=3
)
plt.scatter(range(len(y_test)), y_test.values[sorted_idx], alpha=0.6, label='Actual')
plt.xlabel("Test Sample Index")
plt.ylabel("Final Exam Score")
plt.title("Predictions with 95% Confidence Intervals")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_intervals.png', dpi=300)
plt.show()

# ---------------------------
# 📈 Feature Contribution Summary
# ---------------------------

print("\n" + "="*60)
print("📈 COMPREHENSIVE FEATURE SUMMARY")
print("="*60)

# Create summary dataframe
feature_importance_pd = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_,
    'SHAP_Value': mean_shap
}).sort_values('Importance', ascending=False)

print("\n🏆 Top 10 Features (by Gini Importance & SHAP):")
print(feature_importance_pd.head(10).to_string(index=False))

# Save all explanations
joblib.dump({
    'model': model,
    'explainer_shap': explainer_shap,
    # 'lime_explainer': lime_explainer,  # ← REMOVE THIS - can't be pickled
    'feature_names': feature_names,
    'shap_importance': shap_importance
}, 'xai_explanations.pkl')

print("\n" + "="*60)
print("✅ XAI Model Complete!")
print("📁 Saved Files:")
print("   - shap_summary_plot.png (Global feature importance)")
print("   - shap_bar_plot.png (Mean SHAP values)")
print("   - shap_waterfall.png (Single prediction explanation)")
print("   - lime_explanation_sample_*.html (Interactive LIME explanations)")
print("   - lime_explanation_sample_*.png (LIME bar charts)")
print("   - pdp_plots.png (Partial dependence plots)")
print("   - shap_interaction.png (Feature interactions)")
print("   - prediction_intervals.png (Uncertainty estimation)")
print("   - xai_explanations.pkl (Model, SHAP explainer & feature names only)")
print("="*60)