import pickle
import numpy as np

def predict_churn(age, gender, plan, tenure, usage):

    with open('models/knn_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('models/le_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)

    with open('models/le_plan.pkl', 'rb') as f:
        le_plan = pickle.load(f)

    
    gender_encoded = le_gender.transform([gender])[0]
    plan_encoded = le_plan.transform([plan])[0]

    
    data = np.array([[age, gender_encoded, plan_encoded, tenure, usage]])

    
    data_scaled = scaler.transform(data)

  
    prediction = model.predict(data_scaled)[0]

    return prediction


if __name__ == "__main__":
    result = predict_churn(25, "Male", "Postpaid", 12, 300)
    print("Prediction:", result)

