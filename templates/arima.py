import pmdarima as pm
import logging
import multiprocessing
from itertools import repeat
from typing import Tuple

import numpy as np
import pandas as pd
import pmdarima as pm
from prophet import Prophet

from src.config import (
    TRANSACTION_DATE,
    DS,
    Y,
    YHAT,
    LEVEL,
)

pd.options.mode.chained_assignment = None

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train_single_arima_model(
    group: Tuple, train_df: pd.DataFrame, caps: dict, floors: dict
) -> Tuple[str, Prophet]:
    """
    Training a Prophet model for a single group at a time.
    Identifies the regressors to add by:
    - including all regressors that have the string 'flag', 'marketing' or the corridor's name in them
    Args:
        group: name of the target column (i.e. a group)
        df: Full dataframe with the date series, the targets and the regressors.
        caps: dictionary of the caps per group
        floors: dictionary of the floors per group
    Returns: the group name and a fitted model
    """
    logging.info("Training model {}...".format(group))

    train_df = train_df[train_df["level"] == group[0]]
    train_df = train_df[train_df["kpi"] == group[1]]

    train_df = train_df.rename({TRANSACTION_DATE: DS}, axis=1)
    train_df = train_df.reset_index(drop=True)

    if ((train_df[Y] > 0).sum()) < 10:
        model = "na"
        logging.info(
            "Model {} cannot be trained because there are not enough data points".format(
                group
            )
        )
    else:
        first_nonzero = train_df[Y].ne(0.0).idxmax()
        train_df = train_df.iloc[first_nonzero:]

        train_df = train_df.fillna(1)
        train_df = train_df.set_index(DS)

        model = pm.auto_arima(
            train_df[Y],
            start_p=1,
            start_q=1,
            test="adf",  # use adftest to find optimal 'd'
            max_p=2,
            max_q=2,  # maximum p and q
            m=7,  # frequency of series
            d=None,  # let model determine 'd'
            seasonal=True,
            start_P=0,
            D=0,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )

        print(model.summary())

        model.fit(train_df[Y])

    return group, model


def get_single_arima_prediction(
    model_name: Tuple,
    model_dict: dict,
    future_df: pd.DataFrame,
    caps: dict,
    floors: dict,
):
    """
    Creates the forecast for the forecast horizon from a single model.
    Args:
        model_name: name of the KPI/correspondent/corridor
        model_dict: dictionary of fitted models
        future_df: forecast horizon with flags
        caps: dictionary of the caps per group
        floors: dictionary of the floors per group
    Returns: predictions for the forecast horizon
    """

    model = model_dict[model_name]

    if model == "na":
        logging.info("Could not predict {} ".format(model_name))
        future_forecast = future_df[[DS]]
        future_forecast[0] = 0.0
        future_forecast[["level", "kpi"]] = model_name
    else:

        future_forecast, CI = model.predict(
            n_periods=len(future_df[[DS]]), return_conf_int=True, alpha=0.1
        )
        future_forecast = pd.DataFrame(future_forecast)
        future_forecast[["lower_CI_rec_curr", "upper_CI_rec_curr"]] = CI
        future_forecast.index = future_df[DS]
        future_forecast[["level", "kpi"]] = model_name
        future_forecast.reset_index(inplace=True)
        logging.info("Predictions done for {}".format(model_name))

    return future_forecast


def get_predictions(
    models_dict: dict, future_df: pd.DataFrame, caps: dict, floors: dict
) -> pd.DataFrame:
    """
    Produces the predictions for the forecast horizon with each model.
    Args:
        models_dict: kpi and model pair
        future_df: time series dataset with dates for the forecast horizon and the regressors
        caps: dictionary of the caps per group
        floors: dictionary of the floors per group
    Returns: full predictions
    """

    with multiprocessing.Pool(4) as p:
        df = pd.concat(
            p.starmap(
                get_single_arima_prediction,
                zip(
                    models_dict,
                    repeat(models_dict),
                    repeat(future_df),
                    repeat(caps),
                    repeat(floors),
                ),
            )
        )

    df[YHAT] = np.where(df[YHAT] < 0, 1, df[YHAT])
    return df.reset_index(drop=True)


def train_arima_models(df: pd.DataFrame, caps: dict, floors: dict) -> dict:
    """
    Trains a Prophet model for each KPI.
    Args:
        df: Full dataframe with the date series, the targets and the regressors.
        caps: dictionary of the caps per group
        floors: dictionary of the floors per group
        regressors: list of regressors added to the model
    Returns: A dictionary of the KPIs and their fitted models.
    """
    df_groups = df.groupby(["level", "kpi"])["y"].count().index

    with multiprocessing.Pool(8) as p:
        models = p.starmap(
            train_single_arima_model,
            zip(df_groups, repeat(df), repeat(caps), repeat(floors)),
        )

    return dict(models)


def retrain_odd_models(
    train_df, models_to_retrain, caps, floors, ts_for_preds, df_preds
):

    retrain_df = train_df[train_df[LEVEL].isin(models_to_retrain)]
    re_models = train_arima_models(retrain_df, caps, floors)

    try:
        with multiprocessing.Pool(4) as p:
            df_preds_re = pd.concat(
                p.starmap(
                    get_single_arima_prediction,
                    zip(
                        re_models,
                        repeat(re_models),
                        repeat(ts_for_preds),
                        repeat(caps),
                        repeat(floors),
                    ),
                )
            )
        df_preds_re = df_preds_re.rename({0: YHAT}, axis=1)
        df_preds_re = df_preds_re.fillna(0.0)
        df_preds_first = df_preds[~(df_preds[LEVEL].isin(models_to_retrain))]
        df_preds_first.rename(
            {"yhat_lower": "lower_CI_rec_curr", "yhat_upper": "upper_CI_rec_curr"},
            axis=1,
            inplace=True,
        )

        df_preds = pd.concat([df_preds_first, df_preds_re])
    except:
        logging.info("Correction ARIMA models could not be trained. Reverting back to Python models.")


    return df_preds
