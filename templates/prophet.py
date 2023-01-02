import logging
import multiprocessing
from itertools import repeat
from typing import Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.utilities import regressor_coefficients

from src.config import (
    TRANSACTION_DATE,
    DS,
    Y,
    prophet_holidays,
    YHAT,
    ADD_MARKETING_CAMPAIGNS,
    prophet_country_holidays,
)
from src.data_processing.process_data import add_marketing_campaigns
from src.utils import suppress_stdout_stderr

pd.options.mode.chained_assignment = None

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train_single_model(
    group: Tuple, df: pd.DataFrame, caps: dict, floors: dict, country_level=False
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
    if ADD_MARKETING_CAMPAIGNS:
        train_df = add_marketing_campaigns(df, group)
    else:
        train_df = df.copy()
    regressors = [r for r in train_df.columns if "flag" in r]
    features_to_train_on = regressors + [TRANSACTION_DATE, "y"]
    train_df = train_df[train_df["level"] == group[0]]
    train_df = train_df[train_df["kpi"] == group[1]]

    train_df = train_df[features_to_train_on].rename({TRANSACTION_DATE: DS}, axis=1)
    train_df = train_df.reset_index(drop=True)
    if ((train_df[Y] > 0).sum()) < 10:
        model = Prophet()
        logging.info(
            "Model {} cannot be trained because there are not enough data points".format(
                group
            )
        )
    else:
        first_nonzero = train_df[Y].ne(0.0).idxmax()
        train_df = train_df.iloc[first_nonzero:]
        train_df["cap"] = caps[group] if caps[group] > 1.0 else 2
        train_df["floor"] = floors[group]
        train_df = train_df.fillna(1)
        with suppress_stdout_stderr():
            model = Prophet(
                growth="linear",
                weekly_seasonality=True,
                yearly_seasonality=True,
                holidays=prophet_holidays,
                seasonality_mode="additive"
                if "activation" in group
                else "multiplicative",
                changepoint_range=0.95 if "EUR" in group else 0.8,
                changepoint_prior_scale=0.05,
                interval_width=0.9,
                seasonality_prior_scale=10
            )
            for regressor in regressors:
                model.add_regressor(regressor)
            model.add_seasonality(name="monthly", period=30.5, fourier_order=10)
            country_code = group[0][-2:]
            if country_level and country_code in prophet_country_holidays:
                model.add_country_holidays(country_name=country_code)

            model.fit(train_df)

    return group, model


def get_single_prediction(
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

    if model.history is None:
        logging.info("Could not predict {} ".format(model_name))
        forecast = future_df[[DS]]
        forecast[YHAT] = 0.0
        forecast[["level", "kpi"]] = model_name
    else:
        if ADD_MARKETING_CAMPAIGNS:
            future_df = add_marketing_campaigns(
                future_df.rename({DS: TRANSACTION_DATE}, axis=1), model_name
            )
        regressors = list(regressor_coefficients(model)["regressor"])
        global_regressors = [r for r in regressors if "flag" in r]
        price_regressor = [r for r in regressors if "price" in r]
        future_df = future_df[global_regressors + [DS]]
        future_df[price_regressor] = 0
        future_df["floor"] = floors[model_name]
        future_df["cap"] = caps[model_name]
        future_df.fillna(0, inplace=True)

        forecast = model.predict(future_df.rename({TRANSACTION_DATE: DS}, axis=1))
        forecast[["level", "kpi"]] = model_name
        logging.info("Predictions done for {}".format(model_name))

    return forecast


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
                get_single_prediction,
                zip(
                    models_dict,
                    repeat(models_dict),
                    repeat(future_df),
                    repeat(caps),
                    repeat(floors),
                ),
            )
        )

    df[YHAT] = np.where(((df[YHAT] < 0) & (df['kpi']!='gross_revenue_gbp')), 0, df[YHAT])
    return df.reset_index(drop=True)


def train_models(df: pd.DataFrame, caps: dict, floors: dict) -> dict:
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
            train_single_model,
            zip(df_groups, repeat(df), repeat(caps), repeat(floors), repeat(True)),
        )

    return dict(models)
