# 💼 Employee Salary Classification

<img width="1538" height="782" alt="image" src="https://github.com/user-attachments/assets/07c1b60c-dc23-41c3-822e-8396ef5ce085" />

This project predicts whether an employee earns **more than \$50K or less than/equal to \$50K** per year based on demographic and work-related features. It includes:

- Data preprocessing and cleaning
- Model training and evaluation across multiple ML algorithms
- Selection and saving of the best model
- An interactive Streamlit web application for real-time predictions

---


## 🧠 Features Used

The model uses the following features to predict salary class:

- `age`
- `workclass`
- `fnlwgt` (final weight)
- `educational-num` (education level as a number)
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `gender`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `native-country`

---

## 🔍 Model Training

The notebook trains and compares multiple models:

- Logistic Regression
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Gradient Boosting

The best model (based on accuracy) is saved as `best_model.pkl` using `joblib`.

---

## 🚀 Streamlit Web App

You can run the Streamlit frontend with:

```bash
streamlit run app.py

