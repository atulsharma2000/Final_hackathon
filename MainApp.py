from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model (ensure these files exist)
model = joblib.load('model.pkl')  # Replace with your actual model file
preprocess_model = joblib.load('preprocessor.pkl')  # Replace with your actual preprocess model file

@app.route('/')
def home():
    return render_template('MainPrediction.html')  # Your home page HTML file

@app.route('/MainPrediction')
def prediction():
    return render_template('MainPrediction.html')  # Your prediction form HTML file

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    name = request.form['name']
    age = int(request.form['age'])
    category = request.form['category']
    gender = request.form['gender']
    marks = float(request.form['marks'])

    # Prepare features array based on inputs
    features = np.array([[name,age, category, gender, marks]])  # Adjust based on model requirements
    processed_features = preprocess_model.transform(features) 

    # Make prediction
    prediction = model.predict(processed_features)

    return render_template('MainPrediction.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)