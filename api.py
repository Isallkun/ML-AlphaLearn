### '''API.py ya ini guys'''
# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 1912 # use port 1912

   model = load_model('recommender_no_rev.h5') # Load "recommender_no_rev.h5"
    print ('Model loaded')
    model_columns = joblib.load('recommend_columns.pkl') # Load "recommend_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)
