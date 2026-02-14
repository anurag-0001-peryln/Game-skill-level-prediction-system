# 🎮 Game Skill Level Prediction System

A Machine Learning powered web application that predicts a player's gaming skill level based on gameplay performance metrics such as reaction time, accuracy, score, and misses.

The project demonstrates a complete ML pipeline — from data analysis and model training to deployment using Flask API.

---

## 🚀 Features

- Predict player skill level in real-time
- REST API built with Flask
- Logistic Regression multi-class classification model
- Automated preprocessing using Scikit-Learn Pipeline
- Cross-validation evaluation
- Data visualization and correlation analysis
- End-to-end ML workflow (Train → Save → Serve → Predict)

---

## 🧠 Machine Learning Approach

The system uses a **Multinomial Logistic Regression** classifier.

### Input Features
- Average Reaction Time
- Accuracy
- Score
- Misses

### Output
- Predicted Skill Level (Beginner / Intermediate / Expert)

### Pipeline
1. Data Cleaning (Remove duplicates)
2. Train-Test Split (Stratified)
3. Feature Scaling (StandardScaler)
4. Classification (Logistic Regression)
5. Cross-Validation
6. Model Serialization using Joblib

---


---

## ⚙️ Installation
```
1️⃣ Clone the repository
git clone https://github.com/yourusername/game-skill-prediction.git
cd game-skill-prediction

2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

3️⃣ Install dependencies
pip install flask pandas scikit-learn matplotlib seaborn joblib flask-cors

🏋️ Train the Model
python train_model.py

🌐 Run the Web Application
python app.py

Open in browser:
http://127.0.0.1:5000/
