import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Load trained model with error handling
# ---------------------------

def load_model(model_path='student_model.pkl', columns_path='model_columns.pkl'):
    """Load model and training columns with error handling"""
    try:
        model = joblib.load(model_path)
        training_columns = joblib.load(columns_path)
        print("✅ Model loaded successfully!")
        print(f"✅ Model expects {len(training_columns)} features")
        return model, training_columns
    except FileNotFoundError as e:
        print(f"❌ Error: Model files not found - {e}")
        print("Please ensure you've trained the model first using your training script.")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)

model, training_columns = load_model()

# ---------------------------
# Input validation functions
# ---------------------------

def validate_float_input(prompt, min_val=0, max_val=24, default=None):
    """Validate float input within range"""
    while True:
        try:
            value = input(prompt)
            if value == "" and default is not None:
                return default
            value = float(value)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("❌ Please enter a valid number")

def validate_int_input(prompt, min_val=0, max_val=20, default=None):
    """Validate integer input within range"""
    while True:
        try:
            value = input(prompt)
            if value == "" and default is not None:
                return default
            value = int(value)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("❌ Please enter a valid integer")

def validate_categorical(prompt, options, default=None):
    """Validate categorical input"""
    while True:
        value = input(prompt)
        if value == "" and default is not None:
            return default
        if value in options:
            return value
        else:
            print(f"❌ Please choose from: {', '.join(options)}")

# ---------------------------
# Display header
# ---------------------------

print("\n" + "="*50)
print("🎓 STUDENT PERFORMANCE PREDICTOR")
print("="*50)
print("\n📋 Please enter the following information:")
print("-" * 50)

# ---------------------------
# Collect user input with validation
# ---------------------------

# Study habits
study_hours = validate_float_input("📚 Study hours per day (0-12): ", 0, 12, 5.0)
smartphone_usage = validate_float_input("📱 Smartphone usage hours (0-12): ", 0, 12, 3.0)
social_media = validate_float_input("💬 Social media hours (0-10): ", 0, 10, 2.0)
gaming = validate_float_input("🎮 Gaming hours (0-10): ", 0, 10, 1.0)
streaming = validate_float_input("📺 Streaming hours (0-10): ", 0, 10, 1.5)

# Health & lifestyle
sleep_hours = validate_float_input("😴 Sleep hours (4-10): ", 4, 10, 7.0)
exercise_hours = validate_float_input("🏃 Exercise hours (0-5): ", 0, 5, 1.0)
caffeine = validate_int_input("☕ Caffeine intake cups (0-10): ", 0, 10, 1)

# Academic metrics
attendance = validate_float_input("📖 Class attendance percent (0-100): ", 0, 100, 85)
assignment = validate_float_input("✏️ Assignment completion percent (0-100): ", 0, 100, 80)
motivation = validate_float_input("💪 Motivation level (1-10): ", 1, 10, 7)

# Categorical variables
gender = validate_categorical("👤 Gender (Male/Female/Other): ", ['Male', 'Female', 'Other'], 'Male')
mental_health = validate_categorical("🧠 Mental health status (Good/Average/Poor): ", 
                                     ['Good', 'Average', 'Poor'], 'Average')
internet_quality = validate_categorical("🌐 Internet quality (Good/Average/Poor): ", 
                                        ['Good', 'Average', 'Poor'], 'Good')

# Optional: Actual score for comparison
print("\n" + "-" * 50)
include_actual = input("Do you want to enter actual exam score for comparison? (y/n): ").lower()
actual_score = None
if include_actual == 'y':
    actual_score = validate_float_input("🎯 Actual exam score (0-100): ", 0, 100)

# ---------------------------
# Create DataFrame
# ---------------------------

new_data = pd.DataFrame([{
    'gender': gender,
    'study_hours_per_day': study_hours,
    'smartphone_usage_hours': smartphone_usage,
    'social_media_hours': social_media,
    'gaming_hours': gaming,
    'streaming_hours': streaming,
    'sleep_hours': sleep_hours,
    'exercise_hours': exercise_hours,
    'class_attendance_percent': attendance,
    'assignment_completion_percent': assignment,
    'caffeine_intake_cups': caffeine,
    'mental_health_status': mental_health,
    'internet_quality': internet_quality,
    'motivation_level': motivation
}])

print("\n🔄 Processing data...")

# ---------------------------
# Feature Engineering (Same as training)
# ---------------------------

# Study efficiency
new_data['study_efficiency'] = np.clip(
    new_data['study_hours_per_day'] / (new_data['sleep_hours'] + 1e-5),
    0, 10
)

# Focus ratio
new_data['focus_ratio'] = np.clip(
    new_data['study_hours_per_day'] / (new_data['smartphone_usage_hours'] + 1e-5),
    0, 20
)

# Total screen time
new_data['total_screen_time'] = (
    new_data['social_media_hours'] +
    new_data['gaming_hours'] +
    new_data['streaming_hours']
)

# Distraction ratio
new_data['distraction_ratio'] = new_data['total_screen_time'] / (new_data['study_hours_per_day'] + 1e-5)

# Health metrics
new_data['health_score'] = new_data['sleep_hours'] + new_data['exercise_hours']
new_data['stress_proxy'] = new_data['caffeine_intake_cups'] / (new_data['sleep_hours'] + 1e-5)

# Engagement score
new_data['engagement_score'] = (
    new_data['class_attendance_percent'] +
    new_data['assignment_completion_percent']
) / 2

# Balance score
new_data['balance_score'] = new_data['study_hours_per_day'] / (
    new_data['total_screen_time'] + new_data['sleep_hours'] + 1e-5
)

# Motivation effect
new_data['motivation_effect'] = new_data['motivation_level'] * new_data['study_hours_per_day']

# Study vs screen ratio
new_data['study_vs_screen'] = new_data['study_hours_per_day'] / (new_data['total_screen_time'] + 1e-5)

# Sleep efficiency
new_data['sleep_efficiency'] = new_data['sleep_hours'] * new_data['study_efficiency']

# Mental pressure
new_data['mental_pressure'] = new_data['stress_proxy'] * new_data['distraction_ratio']

# Screen per study hour
new_data['screen_per_study_hour'] = new_data['total_screen_time'] / (new_data['study_hours_per_day'] + 1e-5)

# ---------------------------
# Drop redundant columns (Same as training)
# ---------------------------

new_data = new_data.drop(columns=[
    'class_attendance_percent',
    'gaming_hours',
    'streaming_hours',
    'smartphone_usage_hours',
    'social_media_hours'
], errors='ignore')

# ---------------------------
# Encode categorical variables
# ---------------------------

new_data = pd.get_dummies(new_data, drop_first=True)

# ---------------------------
# Align with training columns
# ---------------------------

# Add missing columns with 0
for col in training_columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Remove extra columns
new_data = new_data[training_columns]

# ---------------------------
# Make prediction
# ---------------------------

prediction = model.predict(new_data)[0]
prediction = np.clip(prediction, 0, 100)  # Ensure score is within 0-100

# ---------------------------
# Display results
# ---------------------------

print("\n" + "="*50)
print("📊 PREDICTION RESULTS")
print("="*50)

def get_grade(score):
    """Convert score to letter grade"""
    if score >= 90:
        return "A", "🌟 Excellent", "#2ecc71"
    elif score >= 80:
        return "B", "👍 Good", "#3498db"
    elif score >= 70:
        return "C", "📘 Satisfactory", "#f39c12"
    elif score >= 60:
        return "D", "⚠️ Needs Improvement", "#e67e22"
    else:
        return "F", "❗ At Risk", "#e74c3c"

grade_letter, grade_desc, _ = get_grade(prediction)

print(f"\n🎯 Predicted Exam Score: {prediction:.1f}/100")
print(f"📚 Grade: {grade_letter} - {grade_desc}")

# Show comparison if actual score provided
if actual_score is not None:
    print("\n" + "-" * 30)
    print("📈 COMPARISON:")
    print(f"   Actual Score: {actual_score:.1f}/100")
    print(f"   Predicted Score: {prediction:.1f}/100")
    
    difference = prediction - actual_score
    error = abs(difference)
    error_percentage = (error / actual_score) * 100 if actual_score > 0 else 0
    
    print(f"   Difference: {difference:+.1f} points")
    print(f"   Absolute Error: {error:.1f} points")
    print(f"   Error Percentage: {error_percentage:.1f}%")
    
    # Accuracy assessment
    if error_percentage <= 5:
        accuracy_msg = "🎯 Excellent! Prediction is very accurate!"
    elif error_percentage <= 10:
        accuracy_msg = "✅ Good prediction accuracy"
    elif error_percentage <= 20:
        accuracy_msg = "⚠️ Moderate accuracy - consider more data"
    else:
        accuracy_msg = "📝 Low accuracy - unusual student pattern detected"
    
    print(f"   {accuracy_msg}")

# ---------------------------
# Provide recommendations
# ---------------------------

print("\n" + "="*50)
print("💡 PERSONALIZED RECOMMENDATIONS")
print("="*50)

recommendations = []

# Study habits
if study_hours < 4:
    recommendations.append("📚 Increase study hours to 4-6 hours daily for better performance")
elif study_hours > 10:
    recommendations.append("⚠️ Very high study hours - ensure you're taking adequate breaks")

# Sleep
if sleep_hours < 6:
    recommendations.append("😴 Increase sleep to 7-8 hours - sleep is crucial for memory consolidation")
elif sleep_hours > 9:
    recommendations.append("⏰ Consider reducing sleep slightly if feeling lethargic")

# Screen time
total_screen = social_media + gaming + streaming
if total_screen > 6:
    recommendations.append("📱 Reduce total screen time - currently high at {:.1f} hours".format(total_screen))
elif total_screen < 2:
    recommendations.append("👍 Good screen time management!")

# Social media specific
if social_media > 3:
    recommendations.append("💬 Consider limiting social media to 2 hours - high usage correlates with lower grades")

# Exercise
if exercise_hours < 0.5:
    recommendations.append("🏃 Add some physical activity - even 15-30 minutes daily boosts brain function")

# Attendance
if attendance < 75:
    recommendations.append("📖 Improve class attendance - regular attendance is strongly correlated with better grades")
elif attendance >= 95:
    recommendations.append("🌟 Excellent attendance! Keep it up!")

# Assignments
if assignment < 80:
    recommendations.append("✏️ Complete all assignments - they're crucial for understanding and grades")

# Caffeine
if caffeine > 3:
    recommendations.append("☕ Reduce caffeine intake - excessive caffeine can disrupt sleep and increase anxiety")

# Motivation
if motivation <= 3:
    recommendations.append("💪 Work on building motivation - set small achievable goals to build momentum")
elif motivation >= 8:
    recommendations.append("🎯 Great motivation level! Channel it effectively with structured study plans")

# Mental health
if mental_health == "Poor":
    recommendations.append("🧘 Prioritize mental health - consider speaking with a counselor or practicing mindfulness")
elif mental_health == "Average" and prediction < 70:
    recommendations.append("💚 Improving mental well-being could positively impact your academic performance")

# Internet quality
if internet_quality == "Poor" and prediction < 70:
    recommendations.append("🌐 Poor internet may affect online learning - explore offline resources or study groups")

# General good habits
if not recommendations:
    recommendations.append("🎉 Great habits! You're on the right track for academic success")
    recommendations.append("📈 Continue maintaining this balanced approach to studying")

for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec}")

# ---------------------------
# Key insights from input
# ---------------------------

print("\n" + "="*50)
print("🔍 KEY INSIGHTS")
print("="*50)

# Calculate some metrics
study_efficiency = study_hours / (sleep_hours + 1)
work_life_balance = "Good" if study_hours <= total_screen + 2 else "Study-heavy"
digital_wellness = "Good" if total_screen <= 4 else "Needs improvement"

print(f"📊 Study Efficiency: {study_efficiency:.2f} (higher is better)")
print(f"⚖️ Work-Life Balance: {work_life_balance}")
print(f"📱 Digital Wellness: {digital_wellness}")
print(f"💪 Health Score: {sleep_hours + exercise_hours:.1f}/15")

# Risk assessment
risk_factors = 0
if study_hours < 3: risk_factors += 1
if sleep_hours < 6: risk_factors += 1
if total_screen > 6: risk_factors += 1
if attendance < 70: risk_factors += 1
if assignment < 70: risk_factors += 1

if risk_factors >= 3:
    print(f"\n⚠️ Academic Risk Level: HIGH ({risk_factors} risk factors)")
    print("   Consider academic support services or counseling")
elif risk_factors >= 1:
    print(f"\n📌 Academic Risk Level: MODERATE ({risk_factors} risk factors)")
    print("   Address the recommendations above to improve")
else:
    print(f"\n✅ Academic Risk Level: LOW")
    print("   You're on track for success!")

# ---------------------------
# Save prediction to history
# ---------------------------

import json
from datetime import datetime

def save_to_history(student_data, prediction, actual_score=None):
    """Save prediction to history file"""
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'student_data': student_data,
        'prediction': float(prediction),
        'actual_score': actual_score if actual_score else None
    }
    
    try:
        # Load existing history
        with open('prediction_history.json', 'r') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    
    # Add new entry
    history.append(history_entry)
    
    # Save back
    with open('prediction_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n💾 Prediction saved to history!")

save_option = input("\n💾 Save this prediction to history? (y/n): ").lower()
if save_option == 'y':
    student_summary = {
        'study_hours': study_hours,
        'sleep_hours': sleep_hours,
        'motivation': motivation,
        'attendance': attendance
    }
    save_to_history(student_summary, prediction, actual_score)

print("\n✨ Prediction complete! Thank you for using Student Performance Predictor.")
print("="*50)