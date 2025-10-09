Customer Churn Prediction using Artificial Neural Networks (ANN)
This project implements a binary classification model using a Deep Neural Network (DNN) built with Keras/TensorFlow to predict whether a bank customer is likely to churn (exit the bank) or not. The solution includes data preprocessing, model training, persistence of preprocessing tools, and a Streamlit web application for real-time prediction.

🚀 Project Goal
The primary objective is to accurately predict customer attrition to enable proactive intervention strategies.

📁 Repository Structure
The core files for this project are organized in the repository root:

ANN-Classification-Churn/
├── app.py                      # 💻 Streamlit web application for real-time prediction.
├── experiments.ipynb           # 🧪 Jupyter Notebook for data exploration, preprocessing, and model training.
├── prediction.ipynb            # 📝 Jupyter Notebook demonstrating model loading and single prediction.
├── Churn_Modelling.csv         # 📊 Original dataset used for training the model.
├── model.h5                    # Saved Keras/TensorFlow model (ANN architecture and weights).
├── scaler.pickle               # Saved StandardScaler object (for numerical feature scaling).
├── label_encoder_gender.pkl    # Saved LabelEncoder object (for 'Gender' column encoding).
├── onehot_encoder_geo.pkl      # Saved OneHotEncoder object (for 'Geography' column encoding).
├── requirements.txt            # Python dependencies needed to run the project.
└── README.md                   # This file.
✨ Model and Methodology
Data Preprocessing
Feature Removal: Irrelevant columns (RowNumber, CustomerId, Surname) were dropped.

Label Encoding: The Gender column was converted to numerical format (0 and 1) using LabelEncoder.

One-Hot Encoding: The Geography column was converted into three binary columns (Geography_France, Geography_Germany, Geography_Spain) using OneHotEncoder.

Scaling: All numerical features were standardized using StandardScaler to prepare the data for the ANN.

ANN Architecture
The model is a Sequential Deep Neural Network with 12 input features and a single output layer:

Input Layer: Expects 12 features.

Hidden Layer 1: 64 neurons, ReLU activation.

Hidden Layer 2: 32 neurons, ReLU activation.

Output Layer: 1 neuron, Sigmoid activation (for binary probability output).

Compilation: Uses the Adam optimizer and binary_crossentropy loss.

Training: Training included EarlyStopping with a patience of 5.

🔧 Setup and Installation
Follow these steps to set up the project environment:

1. Clone the Repository
Bash

git clone https://github.com/ashmanjum04/ANN-Classification-Churn.git
cd ANN-Classification-Churn
2. Create and Activate Virtual Environment
It's highly recommended to use a virtual environment:

Bash

# Using Python's built-in venv
python -m venv churn-env
source churn-env/bin/activate  # On Windows, use: churn-env\Scripts\activate
3. Install Dependencies
Install all necessary libraries using the requirements.txt file:

Bash

pip install -r requirements.txt
4. Verify Model Assets
Ensure the following files are present in the repository root, as they are essential for the app.py script to run:

model.h5

scaler.pickle

label_encoder_gender.pkl

onehot_encoder_geo.pkl

🚀 How to Run the Streamlit Web App
Once the setup is complete and the assets are verified, run the Streamlit application from your terminal:

Bash

streamlit run app.py
This command will launch the application in your web browser, allowing you to input customer data and receive a churn probability prediction.

