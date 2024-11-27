# BuySmart---Purchase-Prediction-App

### Overview
Welcome to **BuySmart**, a user-friendly web app that leverages machine learning to predict customer purchasing behavior. With an intuitive interface and robust backend, this app demonstrates the power of AI in business decision-making.

The app uses a **Logistic Regression model** trained on customer data, with features like `Age`, `Salary`, and `Country`, to predict the purchase behavior (`Yes` or `No`). It also provides visualization for better insights into the model's performance, including a confusion matrix.

![BuySmart App Screenshot](Images/Hero%20page.png)

### Key Features
- **Data Upload**: Users can upload a CSV file of customer data to make predictions.
- **Data Preprocessing**: The app handles missing values, OneHotEncoding of categorical features, and feature scaling.
- **Interactive Model Training**: Users can train a Logistic Regression model with their own data and evaluate its performance.
- **Model Evaluation**: The app provides metrics like accuracy, a confusion matrix, and a classification report.
- **Prediction Interface**: Users can enter demographic information to predict if a customer will make a purchase.

### Tech Stack
- **Python (3.11)**: For data analysis and machine learning model development.
- **Streamlit**: To create an interactive web interface.
- **Pandas, Scikit-learn**: For data preprocessing and model building.
- **Matplotlib, Seaborn**: For data visualization.

## Live Demo

Check out the live app here:

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-green?logo=streamlit)](https://buysmart---purchase-prediction-app-8ywzpcqhjjqi6hi9xxc54s.streamlit.app/)

### Installation and Usage
To run the app locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/purchase-prediction.git


1. Clone the repository:
   ```sh
   git clone https://github.com/stevet97/BuySmart---Purchase-Prediction-App
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
    |-- Architecture/data_preprocessing_+_model_project.py     # Main Streamlit application (Not the
                                                                 catchiest name in the world)
    |-- data/                                                  # Folder for datasets
    |-- requirements.txt                                       # Dependencies for the project
    |-- README.md                                              # Project documentation
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
This project is licensed under the MIT License
MIT License

Copyright (c) [2024] [Stephen Thomas]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

### Contact
For any inquiries, you can reach me at [stephenthomas382@gmail.com]
Let me know if you need help with deploying this README or making further tweaks!


