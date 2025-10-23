==========================================================
# 🧠 Step 2: Train Random Forest model + Plot Predictions
# ==========================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 1️⃣ Create full synthetic data (with all features)
# ----------------------------------------------------------
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'CTR': np.random.uniform(0.01, 1.0, n),
    'CPC': np.random.uniform(1, 10, n),
    'leads': np.random.randint(1, 50, n),
    'keyword_length': np.random.randint(3, 15, n),
    'day_of_week': np.random.randint(0, 6, n),
    'month': np.random.randint(1, 12, n),
    'location': np.random.choice(['delhi', 'mumbai', 'bangalore'], n),
    'device': np.random.choice(['mobile', 'desktop'], n)
})

# ----------------------------------------------------------
# 2️⃣ Generate engagement label with all features involved
# ----------------------------------------------------------
# Add nonlinear + categorical influence for realism
location_effect = data['location'].map({'delhi': 0.05, 'mumbai': 0.02, 'bangalore': 0.07})
device_effect = data['device'].map({'mobile': 0.06, 'desktop': 0.03})

prob = (
    0.4 * data['CTR']**1.5
    - 0.2 * np.log1p(data['CPC'])
    + 0.1 * np.sqrt(data['leads'])
    + 0.05 * np.sin(data['day_of_week'])
    + 0.02 * np.cos(data['month'])
    + location_effect
    + device_effect
    + np.random.normal(0, 0.1, n)
)

# Balanced threshold
threshold = np.percentile(prob, 60)
data['engagement'] = np.where(prob > threshold, 1, 0)

# ----------------------------------------------------------
# 3️⃣ One-hot encode categorical variables
# ----------------------------------------------------------
data_encoded = pd.get_dummies(data, columns=['location', 'device'], drop_first=True)

# Split features/labels
X = data_encoded.drop('engagement', axis=1)
y = data_encoded['engagement']

# ----------------------------------------------------------
# 4️⃣ Train/test split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------
# 5️⃣ Train Random Forest model
# ----------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train, y_train)

# ----------------------------------------------------------
# 6️⃣ Predict probabilities
# ----------------------------------------------------------
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ----------------------------------------------------------
# 7️⃣ Plot predicted engagement probabilities (80% confidence)
# ----------------------------------------------------------
sorted_idx = np.argsort(y_pred_proba)
sorted_proba = y_pred_proba[sorted_idx]
lower = np.percentile(sorted_proba, 10)
upper = np.percentile(sorted_proba, 90)

plt.figure(figsize=(10, 5))
plt.plot(sorted_proba, color='blue', label='Predicted engagement probability')
plt.fill_between(range(len(sorted_proba)), lower, upper, color='blue', alpha=0.2, label='80% confidence zone')
plt.title("Predicted Engagement Probability (with 80% Confidence Zone)")
plt.xlabel("Samples (sorted by predicted probability)")
plt.ylabel("Engagement Probability")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------
# 8️⃣ Save model and columns
# ----------------------------------------------------------
joblib.dump(model, 'model.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')

print("✅ Model trained successfully with all features!")
print("💾 Files saved: model.pkl and model_columns.pkl")
