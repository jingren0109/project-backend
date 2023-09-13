import joblib
import numpy as np
from flask import Flask, Request, request, jsonify
from werkzeug.datastructures import ImmutableOrderedMultiDict
from predict import predict_nn, predict_xgb, predict_loan_grade
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
    info={
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


@app.route("/api/check")
def liveness_check():
    response = jsonify({"result": "OK"})

    # Add the following line to allow any origin to access the resource
    response.headers.add('Access-Control-Allow-Origin', 'https://project-lendwise.netlify.app')

    return response


@app.route("/api/predict/nn")
@swag_from("./swagger/nn.yml", methods=['GET'])
def loan_risk_predict_nn():
    assert isinstance(request.args, ImmutableOrderedMultiDict)

    nn_model_list = [
        "./model/linear/nn_loan_risk_model_0",
        "./model/linear/nn_loan_risk_model_1",
        "./model/linear/nn_loan_risk_model_2",
        "./model/linear/nn_loan_risk_model_3",
        "./model/linear/nn_loan_risk_model_4",
    ]
    data_transformer = "./model/linear/nn_data_transformer.joblib"

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

    description = f"This loan is predicted to recover {round(prediction * 100, 4)}% of its expected return."

    response = jsonify({"value": str(prediction), "description": description})

    # Add the following line to allow any origin to access the resource
    response.headers.add('Access-Control-Allow-Origin', 'https://project-lendwise.netlify.app')

    return response


@app.route("/api/predict/xgb")
@swag_from("./swagger/xgb.yml", methods=['GET'])
def loan_risk_predict_xgb():
    assert isinstance(request.args, ImmutableOrderedMultiDict)

    xgb_model_list = [
        "./model/linear/xgb_loan_risk_model_0.json",
        "./model/linear/xgb_loan_risk_model_1.json",
        "./model/linear/xgb_loan_risk_model_2.json",
        "./model/linear/xgb_loan_risk_model_3.json",
        "./model/linear/xgb_loan_risk_model_4.json",
    ]
    data_transformer = "./model/linear/xgb_data_transformer.joblib"

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

    description = f"This loan is predicted to recover {round(prediction * 100, 4)}% of its expected return."

    response = jsonify({"value": str(prediction), "description": description})

    # Add the following line to allow any origin to access the resource
    response.headers.add('Access-Control-Allow-Origin', 'https://project-lendwise.netlify.app')

    return response


@app.route("/api/predict/ensemble")
def loan_risk_predict_ensemble():
    assert isinstance(request.args, ImmutableOrderedMultiDict)

    nn_model_list = [
        "./model/linear/nn_loan_risk_model_0",
        "./model/linear/nn_loan_risk_model_1",
        "./model/linear/nn_loan_risk_model_2",
        "./model/linear/nn_loan_risk_model_3",
        "./model/linear/nn_loan_risk_model_4",
    ]
    nn_data_transformer = "./model/linear/nn_data_transformer.joblib"

    # Make predictions using each model
    nn_predictions = []
    for i in range(len(nn_model_list)):
        pred = predict_nn(request.args, nn_data_transformer, i, nn_model_list[i])
        nn_predictions.append(pred)

    # Average the predictions across all models
    prediction_nn = np.mean(nn_predictions)

    if prediction_nn < 0:
        logger.warning(f"Predicted value {prediction_nn} is less than 0, setting to 0")
        prediction_nn = 0
    elif prediction_nn > 1:
        logger.warning(f"Predicted value {prediction_nn} is greater than 1, setting to 1")
        prediction_nn = 1

    xgb_model_list = [
        "./model/linear/xgb_loan_risk_model_0.json",
        "./model/linear/xgb_loan_risk_model_1.json",
        "./model/linear/xgb_loan_risk_model_2.json",
        "./model/linear/xgb_loan_risk_model_3.json",
        "./model/linear/xgb_loan_risk_model_4.json",
    ]
    xgb_data_transformer = "./model/linear/xgb_data_transformer.joblib"

    # Make predictions using each model
    xgb_predictions = []
    for i in range(len(xgb_model_list)):
        pred = predict_xgb(request.args, xgb_data_transformer, i, xgb_model_list[i])
        xgb_predictions.append(pred)

    # Average the predictions across all models
    prediction_xgb = np.mean(xgb_predictions)

    if prediction_xgb < 0:
        logger.warning(f"Predicted value {prediction_xgb} is less than 0, setting to 0")
        prediction_xgb = 0
    elif prediction_xgb > 1:
        logger.warning(f"Predicted value {prediction_xgb} is greater than 1, setting to 1")
        prediction_xgb = 1

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

    description = f"This loan is predicted to recover {round(prediction * 100, 4)}% of its expected return."

    response = jsonify({"value": str(prediction), "description": description})

    # Add the following line to allow any origin to access the resource
    response.headers.add('Access-Control-Allow-Origin', 'https://project-lendwise.netlify.app')

    return response


@app.route("/api/predict")
def loan_risk_predict():
    assert isinstance(request.args, ImmutableOrderedMultiDict)

    nn_model_list = [
        "./model/linear/nn_loan_risk_model_0",
        "./model/linear/nn_loan_risk_model_1",
        "./model/linear/nn_loan_risk_model_2",
        "./model/linear/nn_loan_risk_model_3",
        "./model/linear/nn_loan_risk_model_4",
    ]
    nn_data_transformer = "./model/linear/nn_data_transformer.joblib"

    # Make predictions using each model
    nn_predictions = []
    for i in range(len(nn_model_list)):
        pred = predict_nn(request.args, nn_data_transformer, i, nn_model_list[i])
        nn_predictions.append(pred)

    # Average the predictions across all models
    prediction_nn = np.mean(nn_predictions)

    if prediction_nn < 0:
        logger.warning(f"Predicted value {prediction_nn} is less than 0, setting to 0")
        prediction_nn = 0
    elif prediction_nn > 1:
        logger.warning(f"Predicted value {prediction_nn} is greater than 1, setting to 1")
        prediction_nn = 1

    xgb_model_list = [
        "./model/linear/xgb_loan_risk_model_0.json",
        "./model/linear/xgb_loan_risk_model_1.json",
        "./model/linear/xgb_loan_risk_model_2.json",
        "./model/linear/xgb_loan_risk_model_3.json",
        "./model/linear/xgb_loan_risk_model_4.json",
    ]
    xgb_data_transformer = "./model/linear/xgb_data_transformer.joblib"

    # Make predictions using each model
    xgb_predictions = []
    for i in range(len(xgb_model_list)):
        pred = predict_xgb(request.args, xgb_data_transformer, i, xgb_model_list[i])
        xgb_predictions.append(pred)

    # Average the predictions across all models
    prediction_xgb = np.mean(xgb_predictions)

    if prediction_xgb < 0:
        logger.warning(f"Predicted value {prediction_xgb} is less than 0, setting to 0")
        prediction_xgb = 0
    elif prediction_xgb > 1:
        logger.warning(f"Predicted value {prediction_xgb} is greater than 1, setting to 1")
        prediction_xgb = 1

    # Define the weights for each model
    xgb_weight = 0.6
    nn_weight = 0.4

    # Calculate the weighted average of the predictions
    ensemble_prediction = (xgb_weight * prediction_xgb) + (nn_weight * prediction_nn)

    if pd.isna(ensemble_prediction):
        logger.error(f"Prediction is not a valid number: {ensemble_prediction}")
        logger.error(f"Input data: {request.args}")
        return {"error": "There's something wrong with your input."}, 400

    if ensemble_prediction < 0:
        logger.warning(f"Predicted value {ensemble_prediction} is less than 0, setting to 0")
        ensemble_prediction = 0
    elif ensemble_prediction > 1:
        logger.warning(f"Predicted value {ensemble_prediction} is greater than 1, setting to 1")
        ensemble_prediction = 1

    nn_description = f"This loan is predicted to recover {round(prediction_nn * 100, 4)}% of its expected return."
    xgb_description = f"This loan is predicted to recover {round(prediction_xgb * 100, 4)}% of its expected return."
    description = f"This loan is predicted to recover {round(ensemble_prediction * 100, 4)}% of its expected return."

    nn_model_list = [
        "./model/classification/classification/nn_loan_grade_model_0",
        "./model/classification/classification/nn_loan_grade_model_1",
        "./model/classification/classification/nn_loan_grade_model_2",
        "./model/classification/classification/nn_loan_grade_model_3",
        "./model/classification/classification/nn_loan_grade_model_4",
    ]
    nn_data_transformer = "./model/classification/classification/nn_classification_transformer.joblib"

    # Make predictions using each model
    nn_predictions = []
    for i in range(len(nn_model_list)):
        pred = predict_loan_grade(request.args, nn_data_transformer, i, nn_model_list[i], "nn")
        nn_predictions.append(pred)

    print(nn_predictions)

    rf_model_list = [
        "./model/classification/classification/rf_loan_grade_risk_model_0.joblib",
        "./model/classification/classification/rf_loan_grade_risk_model_1.joblib",
        "./model/classification/classification/rf_loan_grade_risk_model_2.joblib",
        "./model/classification/classification/rf_loan_grade_risk_model_3.joblib",
        "./model/classification/classification/rf_loan_grade_risk_model_4.joblib",
    ]
    rf_data_transformer = "./model/classification/classification/rf_classification_transformer.joblib"

    # Make predictions using each model
    rf_predictions = []
    for i in range(len(rf_model_list)):
        pred = predict_loan_grade(request.args, rf_data_transformer, i, rf_model_list[i], "rf")
        rf_predictions.append(pred.tolist()[0])

    print(rf_predictions)

    svm_model_list = [
        "./model/classification/classification/svm_loan_grade_risk_model_0.joblib",
        "./model/classification/classification/svm_loan_grade_risk_model_1.joblib",
        "./model/classification/classification/svm_loan_grade_risk_model_2.joblib",
        "./model/classification/classification/svm_loan_grade_risk_model_3.joblib",
        "./model/classification/classification/svm_loan_grade_risk_model_4.joblib",
    ]
    svm_data_transformer = "./model/classification/classification/svm_classification_transformer.joblib"

    # Make predictions using each model
    svm_predictions = []
    for i in range(len(svm_model_list)):
        pred = predict_loan_grade(request.args, svm_data_transformer, i, svm_model_list[i], "svm")
        svm_predictions.append(pred.tolist()[0])

    print(svm_predictions)

    # Use vote method to get final prediction
    from statistics import mode

    all_predictions = nn_predictions + rf_predictions + svm_predictions
    prediction = mode(all_predictions)


    if pd.isna(prediction):
        logger.error(f"Prediction is not a valid number: {prediction}")
        logger.error(f"Input data: {request.args}")
        return {"error": "There's something wrong with your input."}, 400

    # Encode loan grades as integers
    grade_encoder = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
    }

    # Set minimum acceptable percentage of recovery and loan grade
    min_recovery = 0.7
    medium_recovery = 0.8
    min_grade_encoded = 2

    # Check if the ensemble_recovery and predicted grade meet the minimum criteria
    if ensemble_prediction >= min_recovery and int(prediction) <= min_grade_encoded:
        suggestion = "approve"
    elif ensemble_prediction >= medium_recovery and int(prediction) > min_grade_encoded:
        suggestion = "approve with higher interest rate"
    else:
        suggestion = "reject"

    # Add the suggestion to the JSON response
    response = jsonify({"nn": {"value": str(prediction_nn), "description": nn_description},
                        "xgb": {"value": str(prediction_xgb), "description": xgb_description},
                        "ensemble": {"value": str(ensemble_prediction), "description": description},
                        "grade": {"grade": grade_encoder[int(prediction)], "grade_encoded": str(int(prediction))},
                        "suggestion": suggestion})

    # Add the following line to allow any origin to access the resource
    # response.headers.add('Access-Control-Allow-Origin', 'https://project-lendwise.netlify.app')
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route("/api/predict/grade")
def loan_risk_predict_grade():
    assert isinstance(request.args, ImmutableOrderedMultiDict)

    # Check for the required request parameters
    loan_amnt = request.args.get('loan_amnt')
    term = request.args.get('term')
    emp_length = request.args.get('emp_length')
    if not all([loan_amnt, term, emp_length]):
        return {"error": "Missing one or more required parameters."}, 400

    nn_model_list = [
        "./model/classification/classification/nn_loan_grade_model_0",
        "./model/classification/classification/nn_loan_grade_model_1",
        "./model/classification/classification/nn_loan_grade_model_2",
        "./model/classification/classification/nn_loan_grade_model_3",
        "./model/classification/classification/nn_loan_grade_model_4",
    ]
    nn_data_transformer = "./model/classification/classification/nn_classification_transformer.joblib"

    # Make predictions using each model
    nn_predictions = []
    for i in range(len(nn_model_list)):
        pred = predict_loan_grade(request.args, nn_data_transformer, i, nn_model_list[i], "nn")
        nn_predictions.append(pred)

    print(nn_predictions)

    rf_model_list = [
        "./model/classification/classification/rf_loan_grade_risk_model_0.joblib",
        "./model/classification/classification/rf_loan_grade_risk_model_1.joblib",
        "./model/classification/classification/rf_loan_grade_risk_model_2.joblib",
        "./model/classification/classification/rf_loan_grade_risk_model_3.joblib",
        "./model/classification/classification/rf_loan_grade_risk_model_4.joblib",
    ]
    rf_data_transformer = "./model/classification/classification/rf_classification_transformer.joblib"

    # Make predictions using each model
    rf_predictions = []
    for i in range(len(rf_model_list)):
        pred = predict_loan_grade(request.args, rf_data_transformer, i, rf_model_list[i], "rf")
        rf_predictions.append(pred.tolist()[0])

    print(rf_predictions)

    svm_model_list = [
        "./model/classification/classification/svm_loan_grade_risk_model_0.joblib",
        "./model/classification/classification/svm_loan_grade_risk_model_1.joblib",
        "./model/classification/classification/svm_loan_grade_risk_model_2.joblib",
        "./model/classification/classification/svm_loan_grade_risk_model_3.joblib",
        "./model/classification/classification/svm_loan_grade_risk_model_4.joblib",
    ]
    svm_data_transformer = "./model/classification/classification/svm_classification_transformer.joblib"

    # Make predictions using each model
    svm_predictions = []
    for i in range(len(svm_model_list)):
        pred = predict_loan_grade(request.args, svm_data_transformer, i, svm_model_list[i], "svm")
        svm_predictions.append(pred.tolist()[0])

    print(svm_predictions)

    # Use vote method to get final prediction
    from statistics import mode

    all_predictions = nn_predictions + rf_predictions + svm_predictions
    prediction = mode(all_predictions)



    if pd.isna(prediction):
        logger.error(f"Prediction is not a valid number: {prediction}")
        logger.error(f"Input data: {request.args}")
        return {"error": "There's something wrong with your input."}, 400

    # Encode loan grades as integers
    grade_encoder = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
    }

    response = jsonify({"grade": grade_encoder[int(prediction)], "grade_encoded": str(int(prediction))})

    # Add the following line to allow any origin to access the resource
    # response.headers.add('Access-Control-Allow-Origin', 'https://project-lendwise.netlify.app')
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    app.run()
