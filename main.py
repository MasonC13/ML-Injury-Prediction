import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

# Load the data with row limitation
def load_and_prepare_data(injury_file="InjuryRecord_Enhanced.csv", 
                         playlist_file="PlayList_Enhanced.csv",
                         max_playlist_rows=50000):  # Parameter to limit rows
    
    # Load injury data (usually small, so load completely)
    injury_df = pd.read_csv(injury_file)
    
    # Get all play keys associated with injuries
    injury_play_keys = set(injury_df["PlayKey"].tolist())
    
    # Load PlayList data in a way that prioritizes injury-related plays
    # First, get all plays that resulted in injuries
    chunk_size = 10000  # Process in chunks to avoid memory issues
    injury_plays = []
    non_injury_plays = []
    
    # Read playlist file in chunks and separate injury from non-injury plays
    for chunk in pd.read_csv(playlist_file, chunksize=chunk_size):
        # Separate plays with injuries from those without
        injury_chunk = chunk[chunk["PlayKey"].isin(injury_play_keys)]
        non_injury_chunk = chunk[~chunk["PlayKey"].isin(injury_play_keys)]
        
        injury_plays.append(injury_chunk)
        non_injury_plays.append(non_injury_chunk)
    
    # Combine all injury-related plays
    if injury_plays:
        injury_playlist_df = pd.concat(injury_plays)
    else:
        injury_playlist_df = pd.DataFrame()
        
    # Combine all non-injury plays
    if non_injury_plays:
        non_injury_playlist_df = pd.concat(non_injury_plays)
    else:
        non_injury_playlist_df = pd.DataFrame()
    
    # Calculate how many non-injury plays to keep
    non_injury_rows_to_keep = max(0, max_playlist_rows - len(injury_playlist_df))
    
    # Sample from non-injury plays if needed
    if len(non_injury_playlist_df) > non_injury_rows_to_keep and non_injury_rows_to_keep > 0:
        non_injury_playlist_df = non_injury_playlist_df.sample(n=non_injury_rows_to_keep, random_state=42)
    
    # Combine injury and sampled non-injury plays
    playlist_df = pd.concat([injury_playlist_df, non_injury_playlist_df])
    
    print(f"Loaded {len(injury_df)} injury records")
    print(f"Loaded {len(playlist_df)} playlist records (limited from original size)")
    print(f"Of these, {len(injury_playlist_df)} are linked to injuries")
    
    # Merge datasets
    df = pd.merge(playlist_df, injury_df, on="PlayKey", how="left")
    
    # Create target variable
    df["Injured"] = df["DM_M1"].fillna(0).astype(int)
    
    # Drop rows with missing target values
    df = df.dropna(subset=["Injured"])
    
    # Print class distribution
    print("\nClass distribution in dataset:")
    print(df["Injured"].value_counts())
    print(f"Injury rate: {df['Injured'].mean():.4%}\n")
    
    # Feature selection - removed PositionGroup
    features = [
        "RosterPosition", "StadiumType", "FieldType", "Temperature",
        "Weather", "PlayType"
    ]
    
    # Feature engineering
    # Convert Temperature to a more meaningful representation
    df["TemperatureCategory"] = pd.cut(
        df["Temperature"], 
        bins=[-100, 32, 60, 75, 90, 120],
        labels=["Freezing", "Cold", "Moderate", "Warm", "Hot"]
    )
    
    # Add features list
    features.append("TemperatureCategory")
    
    # Add interaction features
    df["Turf_Hot"] = ((df["FieldType"] == "Turf") & (df["TemperatureCategory"] == "Hot")).astype(int)
    df["Cold_Outdoor"] = ((df["TemperatureCategory"].isin(["Freezing", "Cold"])) & 
                         (df["StadiumType"] == "Outdoor")).astype(int)
    
    features.extend(["Turf_Hot", "Cold_Outdoor"])
    
    return df, features

# Create preprocessing pipeline
def create_preprocessor(numerical_cols, categorical_cols):
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor

# Build and evaluate models - Performance optimized
def build_models(X_train, X_test, y_train, y_test, preprocessor, categorical_cols, numerical_cols, fast_mode=True):
    # Dictionary to store models and their results
    models = {}
    
    # Logistic Regression (already fast)
    model_lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000, 
                                  solver='liblinear', C=0.1)
    model_lr.fit(X_train, y_train)
    models["Logistic Regression"] = model_lr
    
    # Use faster models in fast_mode
    if fast_mode:
        # Random Forest with fewer estimators and using all CPU cores
        model_rf = RandomForestClassifier(random_state=42, class_weight='balanced', 
                                        n_estimators=50, max_depth=5, n_jobs=-1)
        model_rf.fit(X_train, y_train)
        models["Random Forest"] = model_rf
        
        # Gradient Boosting with fewer estimators
        model_gb = GradientBoostingClassifier(random_state=42, n_estimators=50, 
                                            learning_rate=0.1, max_depth=3)
        model_gb.fit(X_train, y_train)
        models["Gradient Boosting"] = model_gb
    else:
        # Original settings
        model_rf = RandomForestClassifier(random_state=42, class_weight='balanced', 
                                        n_estimators=200, max_depth=10, n_jobs=-1)
        model_rf.fit(X_train, y_train)
        models["Random Forest"] = model_rf
        
        # Gradient Boosting
        model_gb = GradientBoostingClassifier(random_state=42, n_estimators=200, 
                                            learning_rate=0.05, max_depth=4)
        model_gb.fit(X_train, y_train)
        models["Gradient Boosting"] = model_gb
    
    # Evaluate all models
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            "classification_report": report,
            "auc": auc,
            "model": model,
            "y_pred_proba": y_pred_proba
        }
        
        # Print evaluation
        print(f"\n--- {name} Results ---")
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
    
    # Plot ROC Curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result["y_pred_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.legend()
    plt.savefig("roc_curves.png")
    
    # Choose the best model (by AUC)
    best_model_name = max(results, key=lambda x: results[x]["auc"])
    best_model = results[best_model_name]["model"]
    
    print(f"\nBest Model: {best_model_name} with AUC = {results[best_model_name]['auc']:.4f}")
    
    # Feature importance analysis for the best model
    if best_model_name == "Logistic Regression":
        coefficients = best_model.coef_[0]
        feature_names = (numerical_cols + 
                         preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_cols).tolist())
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coefficients)
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title(f'Top 20 Feature Importances - {best_model_name}')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        
    elif best_model_name in ["Random Forest", "Gradient Boosting"]:
        # Use built-in feature importance instead of SHAP
        if hasattr(best_model, 'feature_importances_'):
            feature_names = (numerical_cols + 
                            preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_cols).tolist())
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
            plt.title(f'Top 20 Feature Importances - {best_model_name}')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
    
    # Save the best model and preprocessor
    dump(best_model, 'best_model.joblib')
    dump(preprocessor, 'preprocessor.joblib')
    
    return best_model, preprocessor, feature_names

# Main function - removed interactive prediction tool
def main():
    # Parameters for optimization
    fast_mode = True  # Set to False for full model training
    max_rows = 50000  # Adjust based on your computer's capabilities
    
    # Load and prepare data with row limitation
    df, features = load_and_prepare_data(max_playlist_rows=max_rows)
    
    # Split into features and target
    X = df[features]
    y = df["Injured"]
    
    # Define numeric and categorical columns
    numerical_cols = ["Temperature"]
    categorical_cols = [col for col in features if col not in numerical_cols]
    
    # Create preprocessor
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    
    # Preprocess the data
    X_processed = preprocessor.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build and evaluate models
    best_model, preprocessor, feature_names = build_models(
        X_train, X_test, y_train, y_test, preprocessor, categorical_cols, numerical_cols, fast_mode
    )
    
    print("\nModel training complete!")
    print("Model and preprocessor saved as 'best_model.joblib' and 'preprocessor.joblib'")
    print("You can now use the Dash app (app.py) for interactive injury risk prediction.")

if __name__ == "__main__":
    main()








