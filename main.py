import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the data
injury_df = pd.read_csv("InjuryRecord.csv")
playlist_df = pd.read_csv("PlayList.csv")
df = pd.merge(playlist_df, injury_df, on="PlayKey", how="left")
df["Injured"] = df["DM_M1"].fillna(0).astype(int)
df = df.dropna(subset=["Injured"])

# Select features
features = [
    "RosterPosition", "StadiumType", "FieldType", "Temperature",
    "Weather", "PlayType", "PositionGroup"
]
X = df[features]
y = df["Injured"]

# Define numeric and categorical columns
numerical_cols = ["Temperature"]
categorical_cols = [col for col in features if col not in numerical_cols]

# Preprocessing pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Preprocess the data
X_processed = preprocessor.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model with class weights balanced
model_lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
model_lr.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = model_lr.predict(X_test)
print(classification_report(y_test, y_pred_lr, zero_division=0))

# Plot the coefficients for the logistic regression model
coefficients = model_lr.coef_[0]
features_with_coeffs = list(zip(numerical_cols + preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_cols), coefficients))

# Sort the features based on their absolute coefficient value
sorted_features = sorted(features_with_coeffs, key=lambda x: abs(x[1]), reverse=True)

# Display sorted features and their corresponding coefficients
for feature, coef in sorted_features:
    print(f"{feature}: {coef}")

# Function to allow trainer to input data
def input_trainer_data():
    """
    This function prompts the trainer to input data and predicts the likelihood of an injury.
    """
    print("Please enter the following information:")
    
    # Get user input for each feature
    temperature = float(input("Temperature (°F): "))
    field_type = input("Field Type (e.g., 'Turf', 'Grass'): ")
    roster_position = input("Roster Position (e.g., 'OL', 'QB', 'RB', etc.): ")
    weather = input("Weather (e.g., 'Clear', 'Rain', 'Snow'): ")
    play_type = input("Play Type (e.g., 'Run', 'Pass'): ")
    position_group = input("Position Group (e.g., 'Offensive Line', 'Defensive Line', etc.): ")
    
    # Create a dictionary of the input data
    input_data = {
        'Temperature': temperature,
        'FieldType': field_type,
        'RosterPosition': roster_position,
        'Weather': weather,
        'PlayType': play_type,
        'PositionGroup': position_group
    }
    
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input data using the same transformer
    input_processed = preprocessor.transform(input_df)
    
    # Make a prediction
    prediction = model_lr.predict(input_processed)
    
    # Display result
    if prediction[0] == 1:
        print("Injury Risk: High")
    else:
        print("Injury Risk: Low")

# Example usage for trainer input
input_trainer_data()







