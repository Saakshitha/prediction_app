import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from sklearn.datasets import load_diabetes

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://saakshitha:saakshi@cluster0.yl20d.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['predictiondatabase']
collection = db['predictioncollection']
# Load models
def load_ice_cream_models():
    with open("polynomial_models.pkl", 'rb') as file:
        models = pickle.load(file)
    return models

def predict_ice_cream(models, degree, data):
    model = models[f'degree_{degree}']
    prediction = model.predict(data)
    return prediction

def visualize_ice_cream(models, degree, data):
    model = models[f'degree_{degree}']
    x = data[['Temperature (°C)']]
    y = data['Ice Cream Sales (units)']
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Temperature (°C)'], data['Ice Cream Sales (units)'], color='blue', label='Data')
    plt.plot(data['Temperature (°C)'], model.predict(x), color='red', label=f'Polynomial degree {degree}')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Ice Cream Sales (units)')
    plt.title('Ice Cream Sales Prediction')
    plt.legend()
    st.pyplot(plt)

def load_diabetes_models():
    ridge_model = pickle.load(open('ridge_model.pkl', 'rb'))
    lasso_model = pickle.load(open('lasso_model.pkl', 'rb'))
    scalar = pickle.load(open('scalar.pkl', 'rb'))
    return ridge_model, lasso_model, scalar

# Main app
def main():
    st.title("Prediction Application")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the Prediction Mode", ["Ice Cream Sale Prediction", "Diabetes Prediction"])

    # Ice Cream Sale Prediction
    if app_mode == "Ice Cream Sale Prediction":
        st.header("Ice Cream Sale Prediction")
        st.write("Enter the temperature to predict ice cream sales")
        temp = st.number_input("Temperature", min_value=0.0, max_value=100.0)
        degree = st.slider("Select polynomial degree", 2, 5, step=1)

        if st.button("Predict Ice Cream Sales"):
            # Load ice cream models
            models = load_ice_cream_models()
            data = np.array([[temp]])
            prediction = predict_ice_cream(models, degree, data)
            st.write(f"Predicted Ice Cream Sales with polynomial degree {degree}: {prediction[0]}")

        if st.button("Visualize Ice Cream Sales"):
            models = load_ice_cream_models()
            st.write("Visualize the data and model predictions")
            data = pd.read_csv('Ice-cream-selling-data-csv.csv')
            visualize_ice_cream(models, degree, data)
        data = {
            "Temperature": temp,
            "Prediction": prediction[0]
        }
        collection.insert_one(data)

    # Diabetes Prediction
    elif app_mode == "Diabetes Prediction":
        st.header("Diabetes Prediction")
        st.write("Enter the features to predict diabetes progression")

        # Input features
        age = st.number_input("Age")
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        bmi = st.number_input("BMI")
        bp = st.number_input("Blood Pressure")
        s1 = st.number_input("S1")
        s2 = st.number_input("S2")
        s3 = st.number_input("S3")
        s4 = st.number_input("S4")
        s5 = st.number_input("S5")
        s6 = st.number_input("S6")

        # Encode sex column
        sex_encoded = 1 if sex == "Male" else 0 if sex == "Female" else -1

        input_data = np.array([[age, sex_encoded, bmi, bp, s1, s2, s3, s4, s5, s6]])

        model_type = st.selectbox("Choose model type for prediction", ["Ridge", "Lasso"])
        scalar_model = st.selectbox("Choose scalar model for prediction", ["StandardScaler"])

        if st.button("Predict Diabetes Progression"):
            # Load diabetes models
            ridge_model, lasso_model, scalar = load_diabetes_models()
            input_data_scaled = scalar.transform(input_data)

            if model_type == "Ridge":
                prediction = ridge_model.predict(input_data_scaled)
                st.write(f"Ridge Model Prediction: {prediction[0]}")
            elif model_type == "Lasso":
                prediction = lasso_model.predict(input_data_scaled)
                st.write(f"Lasso Model Prediction: {prediction[0]}")

        if st.button("Visualize Diabetes Data"):
            st.write("Visualize the data and model predictions")
            diabetes_data = load_diabetes()
            df = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
            df['target'] = diabetes_data.target

            plt.figure(figsize=(10, 6))
            plt.scatter(df['bmi'], df['target'], color='blue', label='Data')
            plt.xlabel('BMI')
            plt.ylabel('Diabetes Progression')
            plt.title('Diabetes Progression Visualization')
            plt.legend()
            st.pyplot(plt)
        data = {
            "Age": age,
            "Sex": sex,
            "BMI": bmi,
            "Blood Pressure": bp,
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "S4": s4,
            "S5": s5,
            "S6": s6,
            "Prediction": prediction[0]
        }
        collection.insert_one(data)
        
if __name__ == "__main__":
    main()
