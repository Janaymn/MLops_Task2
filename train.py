import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

print("Script started")

# Load data
df = pd.read_csv('data/penguins_clean.csv')
print("Data loaded successfully, shape:", df.shape)

# Label Encoding 
le_species = LabelEncoder()
le_island = LabelEncoder()
le_sex = LabelEncoder()

df['species'] = le_species.fit_transform(df['species'])
df['island'] = le_island.fit_transform(df['island'])
df['sex'] = le_sex.fit_transform(df['sex'])
print("Label encoding done")

# Split features/target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print("Data split into X and y")

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
print("Model trained successfully")

# Predictions
preds = model.predict(X)
print("Predictions done")

# Save metrics
acc = accuracy_score(y, preds)
print("Accuracy calculated:", acc)

os.makedirs('outputs', exist_ok=True)
with open('outputs/metrics.json', 'w') as f:
    json.dump({'accuracy': acc}, f)
print("Metrics saved")

# Confusion matrix
cm = confusion_matrix(y, preds, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('outputs/confusion_matrix.png')
print("Confusion matrix saved to as confusion_matrix.png")

print("Script finished successfully")
