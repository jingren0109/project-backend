import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import logging
import xgboost as xgb


def dataPrepare(data):

    for col_name in data:
        data[col_name] = pd.to_numeric(data[col_name], errors="ignore")

    inverse_columns = [col_name for col_name in data if "inv_" in col_name]

    def invert(value):
        if type(value) is str:
            return 0
        else:
            return 1 / 1 if value == 0 else 1 / value

    for col_name in inverse_columns:
        data[col_name] = data[col_name].map(invert)

    for joint_col, indiv_col in zip(
            ["annual_inc_joint", "dti_joint"], ["annual_inc", "dti"]
    ):
        data[joint_col] = [
            indiv_val if type(joint_val) is str else joint_val
            for joint_val, indiv_val in zip(data[joint_col], data[indiv_col])
        ]

def predict_nn(arguments, transformer, index, model):
    try:
        data = pd.DataFrame(arguments, index=[0])
        dataPrepare(data)

        transformer = joblib.load(transformer)[index]
        X = transformer.transform(data)

        model = load_model(model)

        y_pred = model(X).numpy()[0][0]

        logging.info("The prediction result is: ")
        logging.info(y_pred)

        return y_pred

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return np.nan


def predict_xgb(arguments, transformer, index, model_link):
    try:
        data = pd.DataFrame(arguments, index=[0])
        dataPrepare(data)

        transformer = joblib.load(transformer)[index]
        X = transformer.transform(data)

        model = xgb.Booster()
        model.load_model(model_link)

        # Use the predict method on the loaded model object to obtain the predicted probabilities
        y_pred = model.predict(xgb.DMatrix(X))[0]
        logging.info("The prediction result is: ")
        logging.info(y_pred)
        return y_pred

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return np.nan
