# 🍔 Food Delivery Time Prediction

## Overview
A Machine Learning project that predicts estimated food delivery time using a trained Random Forest Regression model.

The project uses a Streamlit web application to collect delivery-related information and generate delivery time predictions.

## Features
- Random Forest Regression model
- Label Encoding for categorical features
- Interactive Streamlit user interface
- Real-time delivery time predictions
- Seven input features

## Input Features
- Distance_km
- Weather
- Traffic_Level
- Time_of_Day
- Vehicle_Type
- Preparation_Time_min
- Courier_Experience_yrs

**Target Variable:** Delivery_Time_min

**Excluded Feature:** Order_ID

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Pickle

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd Food-Delivery-Time-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run main.py
```

## Project Structure

```text
Food-Delivery-Time-Prediction/
├── main.py
├── Food_Delivery_Times.csv
├── optimized_rf_model.pkl
├── label_encoders.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Model
The application uses a pre-trained Random Forest Regression model and saved Label Encoders to predict estimated food delivery time.

## Author
Ekasjot Singh
