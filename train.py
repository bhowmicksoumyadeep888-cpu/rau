import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess

# Load data
df = load_and_preprocess("data/taxi_data.csv")

# Train/test split
X = df[["zone_id", "hour", "day", "month"]]
y = df["demand"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=300)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/model.pkl")

print("Model trained and saved!")
