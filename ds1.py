# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('diabetes.csv') 

print("First 5 rows of the dataset:")
print(df.head()) 
print("\nDataset Info:")
print(df.info())  

# Define features (X) and target (Y)
X = df[['Age', 'BMI', 'Glucose', 'BloodPressure']]  # Adjust columns based on your dataset
y = df['Outcome']  # Adjust 'Outcome' to your actual target column name

# Handle missing values (optional)
df.fillna(df.mean(), inplace=True)  

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display Results
print("\nModel Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Visualize the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted: 0', 'Predicted: 1'], yticklabels=['Actual: 0', 'Actual: 1'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Optional: Display first few predictions (for demonstration)
print("\nFirst 10 predictions (Predicted vs Actual):")
predicted_vs_actual = pd.DataFrame({'Predicted': y_pred[:10], 'Actual': y_test.iloc[:10].values})
print(predicted_vs_actual)
