# Heart Failure Prediction Models

This repository contains Python code for predicting heart failure using two different models: K-Nearest Neighbors (KNN) and Artificial Neural Network (ANN).

## Data Exploration and Visualization

### Section 1: Correlation between Numerical Features and Label (KNN)
- Utilizes histograms to visualize the correlation between numerical features (e.g., age, creatinine_phosphokinase) and the label.
- Data is loaded from 'data.csv' and unnecessary columns are dropped.

### Section 2: Correlation between Categorical Features and Label (ANN)
- Uses heatmaps to visualize the correlation between categorical features (e.g., anaemia, diabetes) and the label.
- Data is loaded from 'data.csv' and unnecessary columns are dropped.

## KNN Model (Section 3)

- Implements the K-Nearest Neighbors (KNN) model for heart failure prediction.
- Data is preprocessed, scaled, and split into training and testing sets.
- Class imbalance is handled using weights.
- KNN classifier is trained and a simple input prediction function is provided.

## ANN Model (Section 4)

- Implements an Artificial Neural Network (ANN) for heart failure prediction.
- Data is preprocessed, scaled, and split into training and testing sets.
- Class imbalance is handled using weights.
- MLPClassifier is trained and a simple input prediction function is provided.

## Instructions for Input and Prediction

1. Run the code for the desired model (KNN or ANN).
2. Input the required values when prompted (e.g., age, anaemia, etc.).
3. Receive a prediction ('Yes' or 'No') based on the trained model.

Feel free to explore and modify the code to suit your specific requirements.
