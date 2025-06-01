
!pip install xgboost scikit-learn seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

df = pd.read_csv('data.csv')
target_col = 'Crop_Type'
categorical_cols = ['Country', 'Region', 'Adaptation_Strategies']
numerical_cols = [col for col in df.columns if col not in categorical_cols + [target_col]]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
target_le = LabelEncoder()
df[target_col] = target_le.fit_transform(df[target_col])
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
sample_weights = y_train.map(dict(enumerate(class_weights)))
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    random_state=42
)
xgb.fit(X_train, y_train, sample_weight=sample_weights)
y_pred = xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")
y_test_labels = target_le.inverse_transform(y_test)
y_pred_labels = target_le.inverse_transform(y_pred)
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred_labels, target_names=target_le.classes_))
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=target_le.classes_)
plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_le.classes_, yticklabels=target_le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
sample_input = X.iloc[[0]].copy()
sample_input[numerical_cols] = scaler.transform(sample_input[numerical_cols])

# Predict
predicted_crop = target_le.inverse_transform(xgb.predict(sample_input))[0]
print("\nPredicted Crop for Sample Input:", predicted_crop)
