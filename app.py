from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/', methods=['POST'])
def predict_lung_cancer():
    # PREPARE DATA
    # dataset = 'data.csv'
    # df = pd.read_csv(dataset)
    #
    # df.info()
    # DATA Prepared

    # READ INPUT
    data = request.get_json()
    param1 = data['param1']
    param2 = data['param2']
    param3 = data['param3']
    param4 = data['param4']
    param5 = data['param5']

    # TRAIN MODEL
    # train, test = train_test_split(df, test_size=0.3)
    #
    # features = ['XPC lys939gly', 'XPC ala499val', 'XPD Arg156Arg', 'XPF 11985', 'XPF Arg415Gln']
    #
    # train_x = train[features]
    # train_y = train.Target
    #
    # test_x = test[features]
    # test_y = test.Target
    #
    # model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    # model.fit(train_x, train_y)
    #
    # prediction = model.predict(test_x)
    # metrics.accuracy_score(prediction, test_y)
    # MODEL TRAINED

    filename = 'lungcancer.sav'
    model = pickle.load(open(filename, 'rb'))

    data = [[param1, param2, param3, param4, param5]]

    test_df = pd.DataFrame(data,
                           columns=['XPC lys939gly', 'XPC ala499val', 'XPD Arg156Arg', 'XPF 11985', 'XPF Arg415Gln'])
    result = model.predict(test_df)
    probability = model.predict_proba(test_df)[:, 1]

    # SAVE MODEL
    # filename = 'lungcancer.sav'
    # pickle.dump(model, open(filename, 'wb'))
    # Model Saved

    return jsonify({'confidence': result.tolist()[0], 'probability': probability.tolist()[0]})


if __name__ == '__main__':
    app.run()
