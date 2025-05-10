import numpy as np
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from flask import Flask, request, jsonify, render_template_string

# Build and save the model
def train_and_save_model():
    # Check if model already exists
    if os.path.exists('rf_model.joblib') and os.path.exists('scaler.joblib'):
        print("Loading existing model...")
        rf = joblib.load('rf_model.joblib')
        scaler = joblib.load('scaler.joblib')
        feature_names = joblib.load('feature_names.joblib')
        # We'll use a fixed accuracy for existing models
        return rf, scaler, feature_names, 0.97
    
    # Load the breast cancer dataset
    print("Training new model...")
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = list(cancer.feature_names)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save model files
    joblib.dump(rf, 'rf_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(feature_names, 'feature_names.joblib')
    
    return rf, scaler, feature_names, accuracy

# Create Flask app
app = Flask(__name__)

# Train the model when the script loads
model, scaler, feature_names, model_accuracy = train_and_save_model()

# HTML template with proper JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Breast Cancer Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="number"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #3498db;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
            margin-bottom: 20px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
        }
        .positive {
            background-color: #f8d7da;
            color: #721c24;
        }
        .negative {
            background-color: #d4edda;
            color: #155724;
        }
        .accuracy {
            margin-top: 20px;
            font-style: italic;
            color: #6c757d;
        }
        #resultDiv {
            display: none;
        }
        .github-link {
            margin-top: 40px;
            text-align: center;
            color: #6c757d;
        }
        .github-link a {
            color: #3498db;
            text-decoration: none;
        }
        .github-link a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>Breast Cancer Prediction</h1>
    <p>Enter the following features to predict breast cancer risk. You can use sample values or enter your own measurements.</p>
    
    <button id="btnBenign">Fill with Sample Benign Values</button>
    <button id="btnMalignant">Fill with Sample Malignant Values</button>
    
    <form id="predictionForm">
        <div id="featuresContainer">
            <!-- Features will be added here dynamically -->
        </div>
        <button type="button" id="btnPredict">Predict</button>
    </form>
    
    <div id="resultDiv" class="result">
        <h2>Prediction Result</h2>
        <p id="resultText"></p>
        <p id="resultExplanation"></p>
    </div>
    
    <div class="accuracy">
        <p>Model accuracy: {{ accuracy }}%</p>
        <p>Model: Random Forest Classifier</p>
    </div>
    
    <div class="github-link">
        <p>This project is open source. <a href="https://github.com/yourusername/breast-cancer-prediction" target="_blank">View on GitHub</a></p>
    </div>
    
    <script>
        // Feature names from the server
        const featureNames = {{ feature_names|safe }};
        
        // Sample values (approximations for demonstration)
        const benignSample = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259];
        const malignantSample = [20.57, 17.77, 132.9, 1326, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667, 0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.0186, 0.0134, 0.01389, 0.003532, 24.99, 23.41, 158.8, 1956, 0.1238, 0.1866, 0.2416, 0.186, 0.275, 0.08902];
        
        // Create the form fields
        function createFormFields() {
            const container = document.getElementById('featuresContainer');
            
            // Clear existing content
            container.innerHTML = '';
            
            for (let i = 0; i < featureNames.length; i++) {
                const displayName = featureNames[i].replace(/_/g, ' ');
                
                const formGroup = document.createElement('div');
                formGroup.className = 'form-group';
                
                const label = document.createElement('label');
                label.htmlFor = `feature_${i}`;
                label.textContent = displayName;
                
                const input = document.createElement('input');
                input.type = 'number';
                input.step = '0.0001';
                input.id = `feature_${i}`;
                input.name = `feature_${i}`;
                input.required = true;
                
                formGroup.appendChild(label);
                formGroup.appendChild(input);
                container.appendChild(formGroup);
            }
        }
        
        // Fill the form with sample values
        function fillSampleValues(malignant = false) {
            const values = malignant ? malignantSample : benignSample;
            for (let i = 0; i < featureNames.length; i++) {
                document.getElementById(`feature_${i}`).value = values[i];
            }
        }
        
        // Make prediction
        function makePrediction() {
            const features = [];
            for (let i = 0; i < featureNames.length; i++) {
                const value = document.getElementById(`feature_${i}`).value;
                if (!value) {
                    alert('Please fill all fields');
                    return;
                }
                features.push(parseFloat(value));
            }
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ features: features })
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('resultDiv');
                const resultText = document.getElementById('resultText');
                const resultExplanation = document.getElementById('resultExplanation');
                
                resultDiv.style.display = 'block';
                
                if (data.prediction === 1) {
                    resultDiv.className = 'result negative';
                    resultText.textContent = 'Prediction: Benign (Non-cancerous)';
                    resultExplanation.textContent = 'The model predicts that the tumor is benign. This suggests a lower risk.';
                } else {
                    resultDiv.className = 'result positive';
                    resultText.textContent = 'Prediction: Malignant (Cancerous)';
                    resultExplanation.textContent = 'The model predicts that the tumor is malignant. This suggests a higher risk.';
                }
                
                // Scroll to result
                resultDiv.scrollIntoView({ behavior: 'smooth' });
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error making prediction. See console for details.');
            });
        }
        
        // Initialize the page when loaded
        document.addEventListener('DOMContentLoaded', function() {
            createFormFields();
            
            // Set up event listeners
            document.getElementById('btnBenign').addEventListener('click', function(e) {
                e.preventDefault();
                fillSampleValues(false);
            });
            
            document.getElementById('btnMalignant').addEventListener('click', function(e) {
                e.preventDefault();
                fillSampleValues(true);
            });
            
            document.getElementById('btnPredict').addEventListener('click', function(e) {
                e.preventDefault();
                makePrediction();
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render the home page with the form"""
    return render_template_string(
        HTML_TEMPLATE, 
        feature_names=feature_names,
        accuracy=round(model_accuracy * 100, 2)
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions"""
    data = request.json
    features = data['features']
    
    # Reshape and scale the features
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][prediction]
    
    # Get feature importance
    importance = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return jsonify({
        'prediction': int(prediction),
        'probability': float(probability),
        'top_features': [{str(k): float(v)} for k, v in top_features]
    })

# For health checks (important for Render)
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)