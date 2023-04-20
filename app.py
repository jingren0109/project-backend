import numpy as np
from flask import Flask, Request, request, jsonify
from werkzeug.datastructures import ImmutableOrderedMultiDict
from predict import predict_nn, predict_xgb
import pandas as pd
from flasgger import Swagger
from flasgger import swag_from
import logging


class OrderedRequest(Request):
    parameter_storage_class = ImmutableOrderedMultiDict


class MyFlask(Flask):
    request_class = OrderedRequest

app = MyFlask(__name__, static_folder="public", template_folder="views")

swagger_template = dict(
info = {
    'title': 'LendWise Loan Risk Prediction APIs',
    'version': '0.1',
    'description': 'APIs for predicting the recovery rate of a loan based on various input parameters.',
    }
)

# Initialize Swagger with the desired configuration options
swagger_config = {
    "specs": [
            {
                "endpoint": "lendwise",
                "route": "/lendwise",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
    "headers": [],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

logger = logging.getLogger(__name__)

@app.route("/api/predict/nn")
@swag_from("./swagger/nn.yml", methods=['GET'])
def loan_risk_predict_nn():
    assert isinstance(request.args, ImmutableOrderedMultiDict)

    nn_model_list = [
        "./model/nn_loan_risk_model_0",
        "./model/nn_loan_risk_model_1",
        "./model/nn_loan_risk_model_2",
        "./model/nn_loan_risk_model_3",
        "./model/nn_loan_risk_model_4",
    ]
    data_transformer = "./model/nn_data_transformer.joblib"

    # Make predictions using each model
    predictions = []
    for i in range(len(nn_model_list)):
        pred = predict_nn(request.args, data_transformer, i, nn_model_list[i])
        predictions.append(pred)

    # Average the predictions across all models
    prediction = np.mean(predictions)

    if pd.isna(prediction):
        logger.error(f"Prediction is not a valid number: {prediction}")
        logger.error(f"Input data: {request.args}")
        return {"error": "There's something wrong with your input."}, 400

    if prediction < 0:
        logger.warning(f"Predicted value {prediction} is less than 0, setting to 0")
        prediction = 0
    elif prediction > 1:
        logger.warning(f"Predicted value {prediction} is greater than 1, setting to 1")
        prediction = 1

    description = f"This loan is predicted to recover {round(prediction * 100, 1)}% of its expected return."

    response = jsonify({"value": str(prediction), "description": description})

    # Add the following line to allow any origin to access the resource
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route("/api/predict/xgb")
@swag_from("./swagger/xgb.yml", methods=['GET'])
def loan_risk_predict_xgb():
    assert isinstance(request.args, ImmutableOrderedMultiDict)

    xgb_model_list = [
        "./model/xgb_loan_risk_model_0.json",
        "./model/xgb_loan_risk_model_1.json",
        "./model/xgb_loan_risk_model_2.json",
        "./model/xgb_loan_risk_model_3.json",
        "./model/xgb_loan_risk_model_4.json",
    ]
    data_transformer = "./model/xgb_data_transformer.joblib"

    # Make predictions using each model
    predictions = []
    for i in range(len(xgb_model_list)):
        pred = predict_xgb(request.args, data_transformer, i, xgb_model_list[i])
        predictions.append(pred)

    # Average the predictions across all models
    prediction = np.mean(predictions)

    if pd.isna(prediction):
        logger.error(f"Prediction is not a valid number: {prediction}")
        logger.error(f"Input data: {request.args}")
        return {"error": "There's something wrong with your input."}, 400

    if prediction < 0:
        logger.warning(f"Predicted value {prediction} is less than 0, setting to 0")
        prediction = 0
    elif prediction > 1:
        logger.warning(f"Predicted value {prediction} is greater than 1, setting to 1")
        prediction = 1

    description = f"This loan is predicted to recover {round(prediction * 100, 1)}% of its expected return."

    response = jsonify({"value": str(prediction), "description": description})

    # Add the following line to allow any origin to access the resource
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route("/api/predict/ensemble")
@swag_from("./swagger/xgb.yml", methods=['GET'])
def loan_risk_predict_ensemble():
    assert isinstance(request.args, ImmutableOrderedMultiDict)

    nn_model_list = [
        "./model/nn_loan_risk_model_0",
        "./model/nn_loan_risk_model_1",
        "./model/nn_loan_risk_model_2",
        "./model/nn_loan_risk_model_3",
        "./model/nn_loan_risk_model_4",
    ]
    nn_data_transformer = "./model/nn_data_transformer.joblib"

    # Make predictions using each model
    nn_predictions = []
    for i in range(len(nn_model_list)):
        pred = predict_nn(request.args, nn_data_transformer, i, nn_model_list[i])
        nn_predictions.append(pred)

    # Average the predictions across all models
    prediction_nn = np.mean(nn_predictions)

    xgb_model_list = [
        "./model/xgb_loan_risk_model_0.json",
        "./model/xgb_loan_risk_model_1.json",
        "./model/xgb_loan_risk_model_2.json",
        "./model/xgb_loan_risk_model_3.json",
        "./model/xgb_loan_risk_model_4.json",
    ]
    xgb_data_transformer = "./model/xgb_data_transformer.joblib"

    # Make predictions using each model
    xgb_predictions = []
    for i in range(len(xgb_model_list)):
        pred = predict_xgb(request.args, xgb_data_transformer, i, xgb_model_list[i])
        xgb_predictions.append(pred)

    # Average the predictions across all models
    prediction_xgb = np.mean(xgb_predictions)

    # Define the weights for each model
    xgb_weight = 0.6
    nn_weight = 0.4

    # Calculate the weighted average of the predictions
    prediction = (xgb_weight * prediction_xgb) + (nn_weight * prediction_nn)

    if pd.isna(prediction):
        logger.error(f"Prediction is not a valid number: {prediction}")
        logger.error(f"Input data: {request.args}")
        return {"error": "There's something wrong with your input."}, 400

    if prediction < 0:
        logger.warning(f"Predicted value {prediction} is less than 0, setting to 0")
        prediction = 0
    elif prediction > 1:
        logger.warning(f"Predicted value {prediction} is greater than 1, setting to 1")
        prediction = 1

    description = f"This loan is predicted to recover {round(prediction * 100, 1)}% of its expected return."

    response = jsonify({"value": str(prediction), "description": description})

    # Add the following line to allow any origin to access the resource
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    app.run()
