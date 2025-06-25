from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('rf_acc_68.pkl')
scaler = joblib.load('normalizer.pkl')  # No space in filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect 10 input features from form fields named feature1 to feature10
        features = [float(request.form[f'feature{i}']) for i in range(1, 11)]
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        result = 'Cirrhosis Detected' if prediction == 1 else 'No Cirrhosis Detected'
    except Exception as e:
        result = f"Error: {str(e)}"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
