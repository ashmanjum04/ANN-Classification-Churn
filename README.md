# 🧠 ANN Classification – Customer Churn Prediction

This project uses an **Artificial Neural Network (ANN)** model to predict **Customer Churn** — i.e., whether a customer will leave a bank or stay — based on various demographic and financial factors.

---

## 📘 Project Overview

Customer churn prediction is a key problem in the banking and telecom sectors. The goal is to classify whether a customer is likely to leave based on parameters such as age, balance, credit score, activity, etc.

This project demonstrates how to build, train, and evaluate an ANN using **Python** and **TensorFlow/Keras**, starting from data preprocessing to final predictions.

---

## 🧩 Features

- Data preprocessing (encoding categorical variables, scaling)
- ANN model creation using **Keras Sequential API**
- Model training and validation
- Performance evaluation using accuracy and confusion matrix
- User input prediction through a simple Python interface

---

## 📂 Project Structure

```
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

---

## 🧠 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| Programming | Python 3.11 |
| Libraries | TensorFlow, Keras, NumPy, Pandas, Scikit-learn, Matplotlib |
| Environment | Jupyter Notebook / VS Code |
| Version Control | Git & GitHub |

---

## ⚙️ Installation & Setup

### 1. Clone this repository
```bash
git clone https://github.com/ashmanjum04/ANN-Classification-Churn.git
cd ANN-Classification-Churn
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # for Windows
# source venv/bin/activate   # for Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook ann_classification.ipynb
```

---

## 🧬 Model Summary

- **Input Layer:** 11 features  
- **Hidden Layers:** 2 fully connected layers (ReLU activation)  
- **Output Layer:** 1 neuron (Sigmoid activation for binary classification)  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy  

---

## 📊 Results

- **Training Accuracy:** ~86%
- **Testing Accuracy:** ~84%
- **Confusion Matrix** used to evaluate precision and recall

---

## 🧑‍💻 Sample Prediction

```python
# Example input for prediction
input_data = {
    'CreditScore': 650,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
```

The model predicts whether the customer will **Exit (1)** or **Stay (0)**.

---

## 📈 Visualization

- Training vs Validation Accuracy
- Confusion Matrix
- Loss and Accuracy curves for model performance analysis

---

## 💡 Learning Highlights

- Understanding of **ANN architecture** and **forward propagation**
- Data preprocessing using **LabelEncoder** and **StandardScaler**
- Handling categorical variables (Gender, Geography)
- Saving and loading trained models (`.h5` files)
- Performing predictions on new user data

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork this repo and submit a pull request.

---

## 🧾 License

This project is open-source under the **MIT License**.

---

## ✨ Author

**👤 Dongri Ashmanjum**  
🎓 B.Sc. Computer Science, Rayalaseema University  
📍 Kurnool, Andhra Pradesh  
💻 Skills: Python, SQL, Machine Learning, Data Analysis  
🔗 [LinkedIn](https://linkedin.com/in/dongri-ashmanjum-92b327355) | [GitHub](https://github.com/ashmanjum04)

---

⭐ **If you like this project, give it a star on GitHub!**
