from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import importlib
try:
    _flask_cors = importlib.import_module('flask_cors')
    CORS = getattr(_flask_cors, 'CORS')
except Exception:
    
    def CORS(app, resources=None, origins="*"):
        return None

app = Flask(__name__)

CORS(app)  

model = joblib.load('skill_model.pkl')

@app.route('/')
def home():
    return render_template('game.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return jsonify({'skill_level': prediction})

if __name__ == '__main__':
    app.run(debug=True)
