from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model_pipeline = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = {
        'balance': float(request.form['balance']),
        'day': int(request.form['day']),
        'campaign': int(request.form['campaign']),
        'duration': int(request.form['duration']),
        #'previous': int(request.form['previous']),
        #'pdays': int(request.form['pdays']),
        'job': request.form['job'],
        'marital': request.form['marital'],
        'education': request.form['education'],
        'default': request.form['default'],
        'housing': request.form['housing'],
        'loan': request.form['loan'],
        'contact': request.form['contact'],
        'month': request.form['month'],
        'poutcome': request.form['poutcome']
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model_pipeline.predict(input_df)
    result = "Yes" if prediction[0] == 1 else "No"

    return render_template('index.html', prediction_text=f"Will the client subscribe to a term deposit? {result}")

if __name__ == '__main__':
    app.run(debug=True)
