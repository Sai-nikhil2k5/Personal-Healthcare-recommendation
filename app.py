from flask import Flask, render_template, request
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

app = Flask(__name__)

model = joblib.load("model.pkl")

if not os.path.exists("static"):
    os.makedirs("static")

X = pd.read_csv("X.csv")
y_test = pd.read_csv("y_test.csv")
X_test = pd.read_csv("X_test.csv")

@app.route('/')
def home():
    generate_confusion_matrix()
    generate_correlation_heatmap()
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = request.form['gender']
        blood_pressure = int(request.form['blood_pressure'])
        cholesterol = int(request.form['cholesterol'])
        heart_rate = int(request.form['heart_rate'])
        glucose = int(request.form['glucose'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status']
        exercise_level = request.form['exercise_level']

        patient_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'blood_pressure': [blood_pressure],
            'cholesterol': [cholesterol],
            'heart_rate': [heart_rate],
            'glucose': [glucose],
            'bmi': [bmi],
            'smoking_status': [smoking_status],
            'exercise_level': [exercise_level]
        })

        prediction = model.predict(patient_data)[0]
        mapping = {
            0: "No action needed - Healthy",
            1: "Regular check-up advised",
            2: "Lifestyle changes required (Diet, Exercise)",
            3: "Medication recommended"
        }

        generate_confusion_matrix()
        generate_correlation_heatmap()
        return render_template('index.html', result=mapping[prediction])

def generate_confusion_matrix():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Action', 'Check-up', 'Lifestyle', 'Medication'],
                yticklabels=['No Action', 'Check-up', 'Lifestyle', 'Medication'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('static/confusion_matrix.png')
    plt.close()

def generate_correlation_heatmap():
    numeric_X = X.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidth=0.5)
    plt.title('Correlation Matrix (Numeric Features)')
    plt.savefig('static/correlation_matrix.png')
    plt.close()


if __name__ == '__main__':
    app.run(debug=True)
