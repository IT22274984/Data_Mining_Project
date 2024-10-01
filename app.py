from flask import Flask, request, render_template, redirect, url_for
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
    # Get form data
    input_data = {
        'age': int(request.form['age']),
        'job': request.form['job'],
        'marital': request.form['marital'],
        'education': request.form['education'],
        'default': request.form['default'],
        'balance': float(request.form['balance']),
        'housing': request.form['housing'],
        'loan': request.form['loan'],
        'contact': request.form['contact'],
        'day': int(request.form['day']),
        'month': request.form['month'],
        'duration': int(request.form['duration']), 
        'campaign': int(request.form['campaign']),
        'poutcome': request.form['poutcome']
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model_pipeline.predict(input_df)
    result = "Yes" if prediction[0] == 1 else "No"

    # Redirect to result page
    return redirect(url_for('result', prediction=result))

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
