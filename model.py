import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
import joblib
from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_halving_search_cv  # <-- Enable experimental
from sklearn.model_selection import HalvingGridSearchCV

# Ignore warnings to keep output clean
warnings.filterwarnings("ignore")

# ========== Load dataset ==========
df = pd.read_csv('livestock_stress_data.csv')

# ========== Exploratory Data Analysis ==========
print("Dataset Overview:")
print(df.head().to_markdown(index=False))
print("\nDataset Statistics:")
print(df.describe().to_markdown())
print("\nClass Distribution:")
print(df['stress_level'].value_counts(normalize=True).to_markdown())

plt.figure(figsize=(15, 12))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.kdeplot(data=df, x=feature, hue='stress_level', fill=True,
                palette=['#1f77b4', '#ff7f0e'], common_norm=False)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300)
plt.show()

# ========== Data Preprocessing ==========
X = df.drop('stress_level', axis=1)
y = df['stress_level']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ========== Model Pipeline ==========
pipeline = imbpipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Hyperparameter grid (slightly trimmed for faster search)
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 15],
    'classifier__min_samples_split': [2, 5],
    'classifier__max_features': ['sqrt'],
    'classifier__bootstrap': [True]
}

# ========== Model Training (HalvingGridSearchCV) ==========
grid_search = HalvingGridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=2,
    factor=2  # How aggressively to cut candidates each iteration
)

grid_search.fit(X_train, y_train)

# ========== Model Evaluation ==========
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(f"\nBest Parameters: {grid_search.best_params_}")
print("\nClassification Metrics:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Stressed']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Normal', 'Stressed'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png', dpi=300)
plt.show()

# ========== Feature Importance Analysis ==========
# Permutation Importance
result = permutation_importance(best_model, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=-1)

sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X.columns[sorted_idx])
plt.title("Permutation Importances (Test set)")
plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=300)
plt.show()

# ========== Save model ==========
joblib.dump(best_model, 'best_stress_model.pkl')
print("\nModel saved as 'best_stress_model.pkl'")

# ========== Prediction Example ==========
sample_data = X.sample(1, random_state=42)
prediction = best_model.predict(sample_data)
probabilities = best_model.predict_proba(sample_data)

print("\nSample Prediction:")
print(f"Input Features:\n{sample_data.to_markdown(index=False)}")
print(f"\nPredicted Class: {'Stressed' if prediction[0] == 1 else 'Normal'}")
print(f"Class Probabilities: Normal - {probabilities[0][0]:.3f}, "
      f"Stressed - {probabilities[0][1]:.3f}")