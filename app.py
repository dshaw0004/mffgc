from flask import Flask, render_template, request

from mffgc_model import Model

app = Flask(__name__, template_folder='template', static_folder='static')

ML_model = Model()


@app.route('/')
def index():
    return render_template('index.html', result=[False], predicted=False)


@app.route('/style.css')
def stylesheet():
    return render_template('style.css')

@app.route('/marvelfuturefight.png')
def mfflogo():
    return render_template('marvelfuturefight.png')


@app.route('/predict', methods=['POST'])
def predict():
    inputdata = {
        'native_sex': int(request.form.get('native_sex')),
        'sex': int(request.form.get('sex')),
        'native_type': int(request.form.get('native_type')),
        'type': int(request.form.get('type')),
        'native_side': int(request.form.get('native_side')),
        'side': int(request.form.get('side')),
        'native_tier': int(request.form.get('native_tier')),
        'target_tier': float(request.form.get('target_tier')),
        'is_premium': 1 if request.form.get('is_premium') == 'on' else 0,
        'is_extra_cost': 1 if request.form.get('is_extra_cost') == 'on' else 0
    }
    prediction = ML_model.Predict(inputdata)[0]
    # print(ML_model.Predict(inputdata))
    res = [int(ans) for ans in prediction]
    return render_template('index.html', result=res, predicted=True)
