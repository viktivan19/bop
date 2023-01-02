import argparse
import logging
import multiprocessing
import os
from datetime import datetime, timedelta
from itertools import repeat
import pandas as pd
import numpy as np
import psycopg2
from dateutil.relativedelta import relativedelta
import boto3

from src.config import (
    TRANSACTION_DATE,
    DS,
    LEVEL,
    Y,
    KPI,
    YHAT,
    countries_for_arima,
    SQL_FILE_PATH,
)
from src.data_processing.metrics import calculate_receive_country_metrics
from src.data_processing.process_data import (
    check_positive,
    read_sql_query,
    add_flags,
    split_train_and_test,
    get_empty_ts_for_predictions,
    check_forecast_quality, buffer_calculation, clean_pay_methods
)
from src.data_processing.write_data import execute_mogrify
from src.modelling.arima_model import (
    train_arima_models,
    get_single_arima_prediction,
    retrain_odd_models,
)
from src.modelling.prophet_model import train_models, get_single_prediction

keepalive_kwargs = {
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 5,
    "keepalives_count": 5,
}

if 'DB_NAME' in os.environ:
    conn = psycopg2.connect(
        dbname=os.environ["DB_NAME"],
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"],
        **keepalive_kwargs
    )
else:
    ssm = boto3.client("ssm", region_name='eu-west-1')
    conn = psycopg2.connect(
        dbname=os.environ.get('DB_NAME', ssm.get_parameter(Name='/data-science/redshift_db_name', WithDecryption=True)['Parameter']['Value']),
        host=os.environ.get('DB_HOST', ssm.get_parameter(Name='/data-science/redshift_host', WithDecryption=True)['Parameter']['Value']),
        port=os.environ.get('DB_PORT', int(ssm.get_parameter(Name='/data-science/redshift_port', WithDecryption=True)['Parameter']['Value'])),
        user=os.environ.get('DB_USER', ssm.get_parameter(Name='/data-science/redshift_db_user', WithDecryption=True)['Parameter']['Value']),
        password=os.environ.get('DB_PASS', ssm.get_parameter(Name='/data-science/redshift_db_pass', WithDecryption=True)['Parameter']['Value']),
        **keepalive_kwargs
    )


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

parser = argparse.ArgumentParser(description="Receive Country Forecast")

parser.add_argument(
    "--num_days_to_test",
    default=0,
    type=check_positive,
    help="Number of days to test the model on. If 0, the forecast is produced for the next 3 months.",
)
parser.add_argument(
    "--forecast_horizon_weeks",
    default=16,
    type=check_positive,
    help="Number of weeks to forecast ahead.",
)


def model_runner(num_days_in_test: int, forecast_horizon_weeks: int) -> None:
    logging.info("WR Receive Country Forecast STARTED.")

    TODAY = datetime.today().date()
    TRAIN_END = pd.to_datetime(
        pd.to_datetime(TODAY - relativedelta(days=num_days_in_test))
    )
    TRAIN_START = TRAIN_END - relativedelta(years=3)
    FORECAST_END = TODAY + relativedelta(weeks=forecast_horizon_weeks)

    # Read in the data
    transactions_data = read_sql_query(
        SQL_FILE_PATH + "worldremit_receive_country.SQL", conn, TRAIN_START
    )
    transactions_data[LEVEL] = (
        transactions_data["country_name"]
        + "_"
        + transactions_data["receive_country_code"]
        + "_"
        + transactions_data["payout_method"]
    )

    transactions_data = transactions_data.rename(
        {"date": TRANSACTION_DATE, "usd_amount": Y}, axis=1
    )

    transactions_data[KPI] = "usd_amount"

    transactions_data[TRANSACTION_DATE] = pd.to_datetime(
        transactions_data[TRANSACTION_DATE]
    )
    transactions_data = transactions_data.drop(
        ["correspondent_schedule", "currency_code", "country_name"], axis=1
    )

    # Aggregate to get daily volumes
    df = transactions_data.groupby([TRANSACTION_DATE, LEVEL, KPI]).sum().reset_index()

    df.fillna(0, inplace=True)

    df = add_flags(df)
    df = clean_pay_methods(df)

    train_df, test_df, caps, floors = split_train_and_test(df, TRAIN_END)

    # Create df for predictions
    if num_days_in_test == 0:
        ts_for_preds = get_empty_ts_for_predictions(
            preds_start=TODAY, preds_end=FORECAST_END
        )
    else:
        ts_for_preds = get_empty_ts_for_predictions(
            preds_start=TRAIN_END, preds_end=TODAY
        )

    # Train the SARIMA models
    arima_train_df = train_df[train_df[LEVEL].isin(countries_for_arima)]
    arima_models = train_arima_models(arima_train_df, caps, floors)
    try:
        with multiprocessing.Pool(4) as p:
            df_preds_arima = pd.concat(
                p.starmap(
                    get_single_arima_prediction,
                    zip(
                        arima_models,
                        repeat(arima_models),
                        repeat(ts_for_preds),
                        repeat(caps),
                        repeat(floors),
                    ),
                )
            )
        df_preds_arima.rename({0: YHAT}, axis=1, inplace=True)

    except:
        prophet_models = train_models(arima_train_df, caps, floors)
        with multiprocessing.Pool(4) as p:
            df_preds_arima = pd.concat(
                p.starmap(
                    get_single_prediction,
                    zip(
                        prophet_models,
                        repeat(prophet_models),
                        repeat(ts_for_preds),
                        repeat(caps),
                        repeat(floors),
                    ),
                )
            )


    # Train the Prophet models
    prophet_train_df = train_df[~train_df[LEVEL].isin(countries_for_arima)]
    prophet_models = train_models(prophet_train_df, caps, floors)
    with multiprocessing.Pool(4) as p:
        df_preds_prophet_full = pd.concat(
            p.starmap(
                get_single_prediction,
                zip(
                    prophet_models,
                    repeat(prophet_models),
                    repeat(ts_for_preds),
                    repeat(caps),
                    repeat(floors),
                ),
            )
        )

    df_preds_prophet = df_preds_prophet_full[
        [DS, YHAT, LEVEL, KPI, "yhat_lower", "yhat_upper"]
    ]
    df_preds_prophet.rename(
        {"yhat_lower": "lower_CI_rec_curr", "yhat_upper": "upper_CI_rec_curr"},
        axis=1,
        inplace=True,
    )

    df_preds = pd.concat([df_preds_arima, df_preds_prophet])
    models_to_retrain = check_forecast_quality(df_preds, train_df)

    if models_to_retrain:
        df_preds = retrain_odd_models(
            train_df, models_to_retrain, caps, floors, ts_for_preds, df_preds
        )

    df_preds["yhat"] = np.where(df_preds["yhat"] < 0, 0, df_preds["yhat"])
    df_preds["lower_CI_rec_curr"] = np.where(
        df_preds["lower_CI_rec_curr"] < 0, 0, df_preds["lower_CI_rec_curr"]
    )
    df_preds["upper_CI_rec_curr"] = np.where(
        df_preds["upper_CI_rec_curr"] < 0, 0, df_preds["upper_CI_rec_curr"]
    )

    # Converting the final predictions to USD
    df_preds["run_time"] = pd.Timestamp.now()
    df_preds[["receive_country_name", "receive_country_code", "payout_method"]] = df_preds[
        LEVEL
    ].str.split("_", expand=True)
    full_preds = df_preds.groupby(['ds', 'run_time', 'receive_country_name'])['forecast_usd'].sum().reset_index()
    full_preds = full_preds.rename({"yhat": "forecast_usd"}, axis=1)
    full_preds = full_preds.drop(["kpi", "level"], axis=1)

    last_week_actuals = train_df[train_df['transaction_date'] >= pd.to_datetime(TRAIN_END - timedelta(days=7))]
    last_week_actuals[["receive_country_name", "receive_country_code", "payout_method"]] = last_week_actuals[
        LEVEL
    ].str.split("_", expand=True)
    df_buffered = buffer_calculation(full_preds, last_week_actuals, conn, 'WorldRemit', str(TRAIN_END.date()))

    if num_days_in_test > 0:
        df_with_buffer = df_buffered.rename({DS:'ds'},axis=1)

        test_df[["receive_country_name", "receive_country_code", "payout_method"]] = test_df[
            LEVEL
        ].str.split("_", expand=True)
        test_df = test_df.groupby(['transaction_date', 'receive_country_name'])['y'].sum().reset_index()
        test_df = test_df[['transaction_date', 'receive_country_name', 'y']].rename({'transaction_date':'ds'},axis=1)
        df_full = df_with_buffer[['ds', 'receive_country_name', 'forecast_usd', 'forecast_with_buffer']].merge(test_df, on=['ds', 'receive_country_name'])

        df_full['original_abs_error'] = np.abs(df_full['forecast_usd'] - df_full['y'])
        df_full['buffered_abs_error'] = np.abs(df_full['forecast_with_buffer'] - df_full['y'])
        df_full['original_mape'] = df_full['original_abs_error'] / df_full['y']
        df_full['buffered_mape'] = df_full['buffered_abs_error'] / df_full['y']

        df_full = df_full[df_full['ds']< TRAIN_END.date() + pd.DateOffset(days=7)]
        metrics = df_full.groupby('receive_country_name')[['original_mape', 'buffered_mape']].mean().reset_index()

        grouped_metrics = metrics[['original_mape', 'buffered_mape']].mean()
        rank =df_full.groupby("receive_country_name")["y"].sum().sort_values(ascending=False).reset_index().reset_index()[
            ["index", "receive_country_name"]]

        metrics.merge(rank, on='receive_country_name').to_csv('')

    df_buffered = df_buffered.drop('weights', axis=1)

    # Writing the final output to Redshift
    execute_mogrify(df_buffered, "_prv_fx_treasury.wr_receive_country_forecast")

    logging.info("WR Receive Country forecast saved to Redshift.")


if __name__ == "__main__":
    args = parser.parse_args()
    model_runner(args.num_days_to_test, args.forecast_horizon_weeks)
