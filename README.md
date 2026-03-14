# DeepCSAT_Shopzilla
E-commerce Customer Satisfaction Score Prediction using Artificial Neural Network


Overview:-

DeepCSAT is a deep learning project that predicts Customer Satisfaction (CSAT) scores based on customer support interaction data. The model analyzes operational features such as interaction channel, issue category, agent shift, tenure and response time to estimate CSAT scores on a scale of 1–5.

The project demonstrates a complete machine learning workflow including data preprocessing, exploratory data analysis (EDA), model development, evaluation, and deployment using a Streamlit application.

Workflow:-

- Data cleaning and feature engineering
- Exploratory Data Analysis (EDA) to understand interaction patterns and CSAT trends
- Encoding categorical variables and scaling numerical features
- Training a deep neural network using TensorFlow/Keras for multi-class classification
- Evaluating model performance using accuracy, precision, recall and F1-score
- Deploying the trained model through a Streamlit web application for real-time CSAT prediction

Model:-

A neural network was implemented using TensorFlow/Keras with dense layers, ReLU activation, dropout regularization and a Softmax output layer for predicting the five CSAT classes.

Tech Stack:-

Python (3.10.11), Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, Streamlit

Run the App:-

pip install -r requirements.txt
streamlit run app.py
