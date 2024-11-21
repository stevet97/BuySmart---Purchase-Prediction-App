# BuySmart---Purchase-Prediction-App

### Overview
**BuySmart** is a simple yet effective web app built using **Streamlit** that predicts whether a customer will make a purchase based on their demographics and socio-economic factors. The application showcases a typical machine learning workflow, from data preprocessing to model evaluation, and makes the results accessible through an interactive web interface.

The app uses a **Logistic Regression model** trained on customer data, with features like `Age`, `Salary`, and `Country`, to predict the purchase behavior (`Yes` or `No`). It also provides visualization for better insights into the model's performance, including a confusion matrix.

### Key Features
- **Data Upload**: Users can upload a CSV file of customer data to make predictions.
- **Data Preprocessing**: The app handles missing values, OneHotEncoding of categorical features, and feature scaling.
- **Interactive Model Training**: Users can train a Logistic Regression model with their own data, customize some parameters, and see how well the model performs.
- **Model Evaluation**: The app provides metrics like accuracy, a confusion matrix, and a classification report to evaluate model performance.
- **Prediction Interface**: Users can enter demographic information to predict if a customer will make a purchase.

### Tech Stack
- **Python**: For data analysis and machine learning model development.
- **Streamlit**: To create an interactive web interface.
- **Pandas, Scikit-learn**: For data preprocessing and model building.
- **Matplotlib, Seaborn**: For data visualization.

### Installation and Usage
To run the app locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/purchase-prediction.git
   ```

2. Navigate into the project directory:
   ```sh
   cd purchase-prediction
   ```

3. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the app:
   ```sh
   streamlit run app.py
   ```

5. Open the provided URL in your browser to access the app.

### Project Structure
```
|-- purchase-prediction/
    |-- app.py                  # Main Streamlit application
    |-- data/                   # Folder for datasets
    |-- requirements.txt        # Dependencies for the project
    |-- README.md               # Project documentation
```

### Usage Example
1. **Upload Dataset**: Upload your CSV file containing customer data, including columns like `Age`, `Salary`, and `Country`.
2. **Preprocess Data**: Click the "Preprocess Data" button to clean and transform the data for model training.
3. **Train the Model**: Train a Logistic Regression model using your data, and view evaluation metrics like accuracy and confusion matrix.
4. **Make a Prediction**: Enter demographic values to predict if a customer will make a purchase.

### Dataset Information
The app is designed to work with datasets that have the following columns:
- `Country`: Categorical variable representing customer location (e.g., France, Spain, Germany).
- `Age`: Numerical variable representing customer age.
- `Salary`: Numerical variable representing customer income.
- `Purchased`: Target variable (`Yes` or `No`) indicating if the customer made a purchase.

### Future Work
- **Deploying the Web App**: Deploying the app to a cloud platform like **Heroku** or **AWS**.
- **Improving Model Performance**: Trying other models like Random Forest or boosting algorithms for better predictions.
- **User Authentication**: Adding login functionality for personalized usage.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

### Contact
For any inquiries, you can reach me at [your-email@example.com].

