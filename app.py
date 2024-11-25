# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st

# Streamlit Title and Description
st.title("BuySmart: Purchase Prediction App")
st.markdown("""
Welcome to BuySmart! This app predicts whether customers will make a purchase based on their demographic data.  
Steps:
1. Upload a dataset or use the default one.
2. View data preprocessing and model training.
3. Evaluate model performance and make predictions interactively.
""")

# File Upload or Default Dataset
file_path = 'Data/Data.csv'
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.info("Using default dataset.")
    else:
        st.error("Default dataset not found. Please upload a file.")
        st.stop()

# Display Dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.write("**Dataset Summary:**")
st.write(df.describe())
st.write("**Missing Values:**")
st.write(df.isnull().sum())

# Step 1: Handle Missing Values
st.subheader("Data Preprocessing")
st.write("Handling missing values...")
categorical_imputer = SimpleImputer(strategy='most_frequent')
df['Country'] = categorical_imputer.fit_transform(df[['Country']])

numerical_imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = numerical_imputer.fit_transform(df[['Age', 'Salary']])

# Step 2: One-Hot Encoding
st.write("Encoding categorical variables...")
onehot_encoder = OneHotEncoder(sparse=False)
countries_encoded = onehot_encoder.fit_transform(df[['Country']])
country_df = pd.DataFrame(countries_encoded, columns=onehot_encoder.get_feature_names_out(['Country']))
df = pd.concat([df, country_df], axis=1).drop(['Country'], axis=1)

# Step 3: Define Features and Target
st.write("Splitting data into features and target variable...")
x = df.drop('Purchased', axis=1)
y = df['Purchased'].apply(lambda x: 1 if x == 'Yes' else 0)

# Step 4: Scale Numerical Columns
st.write("Scaling numerical features...")
scaler = StandardScaler()
x[['Age', 'Salary']] = scaler.fit_transform(x[['Age', 'Salary']])

# Step 5: Split the Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
st.success("Preprocessing Complete!")

# Step 6: Train Logistic Regression Model
st.subheader("Model Training")
st.write("Training Logistic Regression model...")
model = LogisticRegression()
model.fit(x_train, y_train)
st.success("Model trained successfully!")

# Step 7: Evaluate the Model
st.subheader("Model Evaluation")
st.write("Evaluating the model...")
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")
st.write("**Confusion Matrix:**")
st.dataframe(conf_matrix)

# Confusion Matrix Heatmap
st.write("**Confusion Matrix Heatmap:**")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.write("**Classification Report:**")
st.json(class_report)

# Step 8: Save Preprocessed Data
x_train.to_csv('x_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)

# Step 9: Allow User to Make Predictions
st.subheader("Make a Prediction")
user_country = st.selectbox("Select Country", df.columns[df.columns.str.startswith('Country_')])
user_age = st.slider("Age", 18, 70, 30)
user_salary = st.number_input("Salary", 10000, 150000, 50000)

# Preprocess User Input
user_data = pd.DataFrame([[user_country, user_age, user_salary]], columns=['Country', 'Age', 'Salary'])
user_data[user_country] = 1  # Set selected country to 1
user_data = user_data.fillna(0)  # Fill other countries with 0
user_data[['Age', 'Salary']] = scaler.transform(user_data[['Age', 'Salary']])

# Make Prediction
if st.button("Predict"):
    prediction = model.predict(user_data)
    result = "Yes" if prediction[0] == 1 else "No"
    st.write(f"Prediction: **{result}** (Will the customer purchase?)")
