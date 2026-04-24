import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

os.makedirs('models', exist_ok=True)
df = pd.read_csv('data/knn_telecom.csv')

le_gender = LabelEncoder()
le_plan = LabelEncoder()


df['Gender'] = le_gender.fit_transform(df['Gender'])
df['PlanType'] = le_plan.fit_transform(df['PlanType'])

X = df[['Age', 'Gender', 'PlanType', 'Tenure', 'MonthlyUsage']]
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_scaled, y)

with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/le_gender.pkl', 'wb') as f:
    pickle.dump(le_gender, f)

with open('models/le_plan.pkl', 'wb') as f:
    pickle.dump(le_plan, f)

print("Model trained and saved successfully!")


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")