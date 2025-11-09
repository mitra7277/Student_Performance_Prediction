import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import hashlib
import os

# -----------------------------
# 🔐 Password Hash Function (for user auth in Streamlit later)
# -----------------------------
def hash_pass(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


# -----------------------------
# 📂 Load Dataset
# -----------------------------
DATA_PATH = "student_data_8_features.csv"  # Must contain 8 features + Final_Exam_Score
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded successfully!")
print("Columns in dataset:", df.columns.tolist())


# -----------------------------
# 🔢 Encode Categorical Columns
# -----------------------------
study_env_map = {"Home": 0, "Library": 1, "Cafe": 2}
motivation_map = {"Low": 0, "Medium": 1, "High": 2}

if "Study Environment" in df.columns:
    df["Study Environment"] = df["Study Environment"].map(study_env_map)
elif "Study_Environment" in df.columns:
    df["Study_Environment"] = df["Study_Environment"].map(study_env_map)
else:
    raise KeyError("❌ Missing column: 'Study Environment' or 'Study_Environment' in CSV")

if "Motivation Level" in df.columns:
    df["Motivation Level"] = df["Motivation Level"].map(motivation_map)
elif "Motivation_Level" in df.columns:
    df["Motivation_Level"] = df["Motivation_Level"].map(motivation_map)
else:
    raise KeyError("❌ Missing column: 'Motivation Level' or 'Motivation_Level' in CSV")


# -----------------------------
# 🧹 Clean and Prepare Columns
# -----------------------------
df.columns = [col.replace(" ", "_") for col in df.columns]

# Check for missing values
if df.isnull().sum().any():
    print("⚠️ Missing values found — filling with column mean...")
    df.fillna(df.mean(), inplace=True)

# -----------------------------
# 🎯 Define Features and Target
# -----------------------------
X = df[[
    "Hours_Studied", "Attendance", "Assignments", "Sleep_Hours",
    "Previous_Score", "Internet_Usage", "Study_Environment", "Motivation_Level"
]].values

y = df["Final_Exam_Score"].values


# -----------------------------
# 📊 Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# 🌳 Train Random Forest Model
# -----------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training completed!")


# -----------------------------
# 📈 Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Evaluation Results:")
print(f"   MSE  = {mse:.4f}")
print(f"   RMSE = {rmse:.4f}")
print(f"   R²   = {r2:.4f}")


# -----------------------------
# 🌟 Feature Importance Visualization
# -----------------------------
features = [
    "Hours_Studied", "Attendance", "Assignments", "Sleep_Hours",
    "Previous_Score", "Internet_Usage", "Study_Environment", "Motivation_Level"
]

importances = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")  # Save as image
print("📊 Feature importance chart saved as 'feature_importance.png'.")


# -----------------------------
# 💾 Save Model and Metadata
# -----------------------------
model_package = {
    "model": model,
    "features": features,
    "metrics": {"MSE": mse, "RMSE": rmse, "R2": r2}
}

joblib.dump(model_package, "student_model_rf_8_features.pkl")
print("\n✅ Model saved as 'student_model_rf_8_features.pkl' successfully!")

# -----------------------------
# 📝 Save Training Report
# -----------------------------
with open("training_report.txt", "w") as f:
    f.write("🎓 Student Performance Prediction Model Report\n")
    f.write("==========================================\n")
    f.write(f"Features used: {features}\n\n")
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n\n")
    f.write("Model File: student_model_rf_8_features.pkl\n")
    f.write("Feature Importance Chart: feature_importance.png\n")

print("📝 Training report saved as 'training_report.txt'.")
