from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pymcdm
import pymcdm.methods as mcdm_methods
from pymcdm.helpers import rrankdata
import pandas as pd
import os
import ast

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Placeholder for sentiment scores and reviews dictionary
sentiment_score = {}

def parse_csv(file_path):
    global sentiment_score
    df = pd.read_csv(file_path)

    # Debugging: Print the first few rows and columns of the CSV
    print("CSV Columns:", df.columns.tolist())
    print("CSV Preview:")
    print(df.head())

    required_columns = {'Phone'}
    required_columns.update(df.columns[1:])  # All columns except the first are features

    # Check for missing columns
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing columns in CSV: {missing_columns}")

    sentiment_score.clear()
    for _, row in df.iterrows():
        phone = row['Phone']
        if phone not in sentiment_score:
            sentiment_score[phone] = {}
        for col in df.columns:
            if col != 'Phone':
                values = ast.literal_eval(row[col])
                sentiment_score[phone][col] = values

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join('/tmp', file.filename)
        file.save(file_path)
        try:
            parse_csv(file_path)
        except KeyError as e:
            return str(e), 400  # Return the error message and a 400 status code

        # Store phones and criteria in session
        session['phones'] = list(sentiment_score.keys())
        session['criteria'] = list(next(iter(sentiment_score.values())).keys())
        
        return redirect(url_for('select'))

@app.route('/select', methods=['GET', 'POST'])
def select():
    if request.method == 'POST':
        phones = request.form.getlist('phone')
        criteria = request.form.getlist('criteria')
        return redirect(url_for('results', phones=','.join(phones), criteria=','.join(criteria)))
    
    phones = session.get('phones', [])
    criteria = session.get('criteria', [])
    return render_template('select.html', phones=phones, criteria=criteria)

@app.route('/results')
def results():
    phones = request.args.get('phones').split(',')
    criteria = request.args.get('criteria').split(',')

    # Prepare the decision matrix and weights
    matrix = []
    weights = []
    total_reviews = sum(sentiment_score[phone][crit][1] for phone in phones for crit in criteria if crit in sentiment_score[phone])
    for phone in phones:
        row = []
        for crit in criteria:
            if crit in sentiment_score[phone]:
                row.append(sentiment_score[phone][crit][0])
            else:
                row.append(0)  # Append zero if no data available
        matrix.append(row)
    weights = [sum(sentiment_score[phone][crit][1] for phone in phones if crit in sentiment_score[phone]) / total_reviews if total_reviews else 0 for crit in criteria]
    
    matrix_np = np.array(matrix, dtype='float')
    weights_np = np.array(weights)
    types_np = np.array([1] * len(criteria))  # Assuming all criteria are benefits

    # Perform VIKOR calculation
    mc = mcdm_methods.VIKOR()
    preferences = mc(matrix_np, weights_np, types_np)

    # Sort preferences and assign ranks
    sorted_indices = np.argsort(preferences)
    sorted_preferences = preferences[sorted_indices]
    sorted_phones = np.array(phones)[sorted_indices]

    # Create the results with ranks
    results = [(sorted_phones[i], sorted_preferences[i], i + 1) for i in range(len(sorted_phones))]

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
