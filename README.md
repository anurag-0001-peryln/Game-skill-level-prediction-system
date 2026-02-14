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

## 🏗️ Project Structure

