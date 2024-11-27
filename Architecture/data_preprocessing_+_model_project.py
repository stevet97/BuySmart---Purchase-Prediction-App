import joblib
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

st.markdown("<h2 style='text-align: center; color: blue;'>BuySmart Prediction Dashboard</h2>", unsafe_allow_html=True)
st.title("BuySmart: Purchase Prediction App")

# Ensure the file path is dynamic and works regardless of the execution location
default_file_path = os.path.join(os.path.dirname(__file__), 'Data', 'Data.csv')
st.write(f"Resolved file path: {default_file_path}")


# File Upload or Default Dataset
default_file_path = 'Data/Data.csv'

# File Upload or Default Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
elif os.path.exists(default_file_path):  # Check if the default dataset exists
    df = pd.read_csv(default_file_path)
    st.info("Using the default dataset.")
else:
    st.error("Default dataset not found. Please upload a file to proceed.")
    st.stop()

#Dataset Preview
st.subheader("Dataset Preview")
st.write("First 5 rows of the dataset:")
st.dataframe(df.head())

st.sidebar.subheader("View Options")
if st.sidebar.checkbox("View Raw Data"):
    st.subheader("Raw Dataset")
    st.write(df)

if st.sidebar.checkbox("View Preprocessed Data"):
    st.subheader("Preprocessed Dataset")
    st.write(df)

# Dataset Insights
st.subheader("Data Insights")
st.bar_chart(df['Purchased'].value_counts())

# Age Distribution
st.subheader("Age Distribution")
fig, ax = plt.subplots()
sns.histplot(df['Age'], kde=True, ax=ax, color='blue')
ax.set_title('Age Distribution')
st.pyplot(fig)

# Salary Distribution
st.subheader("Salary Distribution")
fig, ax = plt.subplots()
sns.boxplot(df['Salary'], ax=ax, color='green')
ax.set_title('Salary Distribution')
st.pyplot(fig)

#Missing Values Overview
st.subheader("Missing Values Overview")
st.write("Missing values before imputation:")
st.write(df.isnull().sum())

#Debug Country column
st.write("Inspecting 'Country' column before imputation:")
st.write("Data type:", df['Country'].dtype)
st.write("Unique values:", df['Country'].unique())
st.write("Shape:", df[['Country']].shape)

# Replace missing or invalid values manually
df['Country'] = df['Country'].replace(['nan', '', None, np.nan], 'Unknown')

# Fill missing values in the 'Country' column with the most frequent value (manual approach)
most_frequent_country = df['Country'].mode()[0]  # Get the most frequent value
df['Country'] = df['Country'].fillna(most_frequent_country)  # Replace missing values
st.success(f"Missing values in 'Country' column replaced with '{most_frequent_country}'.")

# Handle missing values in numerical columns ('Age', 'Salary') using mean imputation
numerical_imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = numerical_imputer.fit_transform(df[['Age', 'Salary']])

# Create and fit the scaler during preprocessing
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])
st.success("Missing values in numerical columns ('Age', 'Salary') have been handled.")

# Perform one-hot encoding for the 'Country' column
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Use sparse_output instead of sparse
countries_encoded = onehot_encoder.fit_transform(df[['Country']])
country_df = pd.DataFrame(countries_encoded, columns=onehot_encoder.get_feature_names_out(['Country']))

# Concatenate one-hot encoded columns with the rest of the dataframe and drop the original 'Country' column
df = pd.concat([df, country_df], axis=1).drop(['Country'], axis=1)
st.success("One-hot encoding for the 'Country' column completed.")


x = df.drop('Purchased', axis=1)
y = df['Purchased'].apply(lambda x: 1 if x == 'Yes' else 0)

x[['Age', 'Salary']] = StandardScaler().fit_transform(x[['Age', 'Salary']])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train and Evaluate Model
st.subheader("Model Training and Evaluation")
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix:")
st.write(conf_matrix)

#Confusion Matrix Heatmap
st.subheader("Confusion Matrix Heatmap")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.subheader("Additional Model Metrics")
precision = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision']
recall = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']

st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1-Score: {f1:.2f}")

# Save model
joblib.dump(model, 'logistic_model.pkl')
st.success("Model saved successfully!")

# Prepare user data for prediction
st.subheader("Make Predictions")

# Collect user input
country_options = [col.replace('Country_', '') for col in df.columns if col.startswith('Country_')]
user_country = st.selectbox("Select Country", country_options, key="user_country_selection")
user_age = st.slider("Select Age", min_value=18, max_value=70, value=30, key="user_age_slider")
user_salary = st.number_input("Enter Salary", min_value=10000, max_value=150000, value=50000, key="user_salary_input")

# Prepare the user's input data
user_data = pd.DataFrame([[user_age, user_salary] +
                          [1 if country == user_country else 0 for country in country_options]],
                         columns=['Age', 'Salary'] + [f"Country_{country}" for country in country_options])

# Reuse the fitted scaler to transform user input
user_data[['Age', 'Salary']] = scaler.transform(user_data[['Age', 'Salary']])

# Predict button and display results
if st.button("Predict", key="predict_button"):
    prediction = model.predict(user_data)
    result = "Yes" if prediction[0] == 1 else "No"
    st.write(f"Prediction: **{result}** (Will the customer purchase?)")

# Check for invalid inputs
if user_salary < 10000 or user_salary > 150000:
    st.error("Salary must be between 10,000 and 150,000.")
else:
    user_data = pd.DataFrame([[user_age, user_salary] +
                              [1 if country == user_country else 0 for country in country_options]],
                             columns=['Age', 'Salary'] + [f"Country_{country}" for country in country_options])
    user_data[['Age', 'Salary']] = scaler.transform(user_data[['Age', 'Salary']])

