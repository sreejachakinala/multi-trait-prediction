from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)

# Load the trained model
loaded_model = joblib.load('model.joblib')
vect=joblib.load('vect.joblib')
values = ['DH_Pooled','GFD_Pooled','GNPS_Pooled','GWPS_Pooled','PH_Pooled','GY_Pooled']
# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle predictions

@app.route('/predict', methods=['POST'])
def predict():
    sequence = request.form.get('sequence')

    if not sequence:
        return render_template('index.html', error="Please enter a sequence.")

    predictions = {}

    for col in values:
        # Vectorize the input sequence using the same CountVectorizer
        sequence_vectorized = vect[col].transform([sequence])

        # Use the trained model to make predictions
        prediction = loaded_model[col].predict(sequence_vectorized)
        predictions[col] = prediction[0].item()  # Convert numpy value to Python type

    return render_template('result.html', sequence=sequence, predictions=predictions)


if __name__ == '__main__':
    app.run(debug=True)