#!/usr/bin/env python
# encoding: utf-8
from flask import Flask, request, jsonify, json, g
from ai.main import deviceSelect, loadModel, loadVocab, inference


app = Flask(__name__)


@app.before_request
def load_ai():
    device = deviceSelect()
    field = loadVocab('ai/vocab/TEXT_obj_kaggle_trained_2.pth')
    model = loadModel(
        'ai/model/textTransformer_states_kaggle_trained_2.pth',
        len(field.vocab),
        device
    )
    g.ai = {
        'model': model,
        'field': field,
        'device': device
    }


@app.route('/')
def index():
    return jsonify({'health': 'ok'})


@app.route('/', methods=['POST'])
def validate():
    if (request.data):
        content = request.get_json(silent=True)
        text = content['body']
    else:
        text = request.form['body']
    try:
        outcome = inference(g.ai['model'], g.ai['field'], text, g.ai['device'])
        response = app.response_class(
            response=json.dumps(outcome),
            status=200,
            mimetype='application/json'
        )
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps(str(e)),
            status=500,
            mimetype='application/json'
        )
        return response


if __name__ == '__main__':
    app.run(debug=True)
