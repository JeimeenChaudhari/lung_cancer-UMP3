import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from lightgbm import LGBMClassifier

# 1. LOAD AND CLEAN DATA
print("=" * 60)
print("LOADING AND PREPROCESSING DATA")
print("=" * 60)

# Load the dataset
df = pd.read_csv('dataset_med.csv')
print(f"\nOriginal dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Drop irrelevant and leaky columns
columns_to_drop = ['id', 'end_treatment_date']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
print(f"\nDropped columns: {[col for col in columns_to_drop if col in df.columns]}")

# Convert diagnosis_date to datetime and extract features
if 'diagnosis_date' in df.columns:
    df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
    df['diagnosis_year'] = df['diagnosis_date'].dt.year
    df['diagnosis_month'] = df['diagnosis_date'].dt.month
    df = df.drop(columns=['diagnosis_date'])
    print("Extracted diagnosis_year and diagnosis_month features")

# Handle missing values
print("\nMissing values before imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Impute numerical columns with median
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'survived' in numerical_cols:
    numerical_cols.remove('survived')

for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Impute categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after imputation:")
print(df.isnull().sum().sum())

# 2. PREPROCESSING PIPELINE
print("\n" + "=" * 60)
print("BUILDING PREPROCESSING PIPELINE")
print("=" * 60)

# Separate features and target
X = df.drop(columns=['survived'])
y = df['survived']

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# 3. TRAIN THE MODEL
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Fit preprocessor and transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\nProcessed training set shape: {X_train_processed.shape}")
print(f"Processed test set shape: {X_test_processed.shape}")

# Train LGBMClassifier
lgbm_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    random_state=42,
    verbose=-1
)

print("\nTraining LGBMClassifier...")
lgbm_model.fit(X_train_processed, y_train)
print("Training completed!")

# 4. EVALUATE AND SAVE ARTIFACTS
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Make predictions
y_pred = lgbm_model.predict(X_test_processed)
y_pred_proba = lgbm_model.predict_proba(X_test_processed)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nAccuracy Score: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# Save artifacts
print("\n" + "=" * 60)
print("SAVING MODEL ARTIFACTS")
print("=" * 60)

with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
print("✓ Saved preprocessor.pkl")

with open('lung_cancer_model.pkl', 'wb') as f:
    pickle.dump(lgbm_model, f)
print("✓ Saved lung_cancer_model.pkl")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)