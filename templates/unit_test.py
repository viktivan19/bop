

@pytest.mark.parametrize("input_string, expected_output", [("3", 3), (3, 3), ("0", 0)])
def test_check_positive(input_string, expected_output):
    actual = check_positive(input_string)

    # Test non-negative cases
    assert actual == expected_output
    # Test negative case
    with pytest.raises(argparse.ArgumentTypeError):
        check_positive("-5")


@pytest.mark.parametrize(
    "input_df, type_of_day, column_name, expected_df",
    [
        (
            pd.DataFrame(
                {
                    TRANSACTION_DATE: pd.to_datetime(
                        [
                            "2021-01-01",
                            "2021-01-02",
                            "2021-01-03",
                            "2021-01-04",
                            "2021-01-05",
                            "2021-01-06",
                            "2021-01-07",
                            "2021-01-29",
                        ]
                    )
                }
            ),
            "BM",
            "last_wd",
            pd.DataFrame(
                {
                    TRANSACTION_DATE: pd.to_datetime(
                        [
                            "2021-01-01",
                            "2021-01-02",
                            "2021-01-03",
                            "2021-01-04",
                            "2021-01-05",
                            "2021-01-06",
                            "2021-01-07",
                            "2021-01-29",
                        ]
                    ),
                    "last_wd": [np.NaN] * 7 + [1.0],
                }
            ),
        ),
        (
            pd.DataFrame(
                {
                    TRANSACTION_DATE: pd.to_datetime(
                        [
                            "2021-01-01",
                            "2021-01-02",
                            "2021-01-03",
                            "2021-01-04",
                            "2021-01-05",
                            "2021-01-06",
                            "2021-01-07",
                            "2021-01-29",
                        ]
                    )
                }
            ),
            "BMS",
            "first_wd",
            pd.DataFrame(
                {
                    TRANSACTION_DATE: pd.to_datetime(
                        [
                            "2021-01-01",
                            "2021-01-02",
                            "2021-01-03",
                            "2021-01-04",
                            "2021-01-05",
                            "2021-01-06",
                            "2021-01-07",
                            "2021-01-29",
                        ]
                    ),
                    "first_wd": [1.0] + [np.NaN] * 7,
                }
            ),
        ),
    ],
    ids=["last_working_day", "first_working_day"],
)
def test_create_special_day_flag(input_df, type_of_day, column_name, expected_df):
    first_day = input_df[TRANSACTION_DATE].min()
    last_day = input_df[TRANSACTION_DATE].max()
    actual_df = create_special_day_flag(
        input_df, type_of_day, column_name, first_day, last_day
    )
    assert_frame_equal(actual_df, expected_df)


def test_merge_corridor_data():
    df_global = pd.DataFrame(
        {
            TRANSACTION_DATE: pd.to_datetime(["2020-03-01", "2020-03-02"]),
            "a": [1, 2],
            "b": [3, 4],
        }
    )
    df_corridor = pd.DataFrame(
        {
            TRANSACTION_DATE: pd.to_datetime(["2020-03-01", "2020-03-02"] * 2),
            CLIENT_COUNTRY: ["country1", "country1", "country2", "country2"],
            RECIPIENT_COUNTRY: ["r1", "r1", "r2", "r2"],
            PAYOUT_METHOD: ["x", "x", "y", "y"],
            "gross_revenue_gbp": [1000, 2000, 3000, 4000],
            "gross_send_amount_send_curr": [100, 200, 300, 200],
        }
    )
    expected_df = pd.DataFrame(
        {
            TRANSACTION_DATE: pd.to_datetime(["2020-03-01", "2020-03-02"]),
            "a": [1, 2],
            "b": [3, 4],
            "country1_r1_x": [100, 200],
            "country2_r2_y": [300, 200],
        }
    )
    actual_df = merge_corridor_data(df_global, df_corridor)

    assert_frame_equal(actual_df, expected_df)


def test_add_flags():
    input_df = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(
                ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]
            )
        }
    )
    expected_result = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(
                ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]
            ),
            "last_working_day_of_month_flag": [0.0, 0.0, 0.0, 0.0],
            "first_working_day_of_month_flag": [1.0, 0, 0, 0],
            "new_year_day_flag": [1.0, 0, 0, 0],
            "last_friday_of_month_flag": [0.0, 0.0, 0.0, 0.0],
        }
    )

    actual_result = add_flags(input_df)
    assert_frame_equal(actual_result, expected_result)


@pytest.mark.parametrize(
    "input_df, train_end, train_expected, test_expected, caps_expected, floors_expected",
    [
        (
            pd.DataFrame(
                {
                    "transaction_date": pd.to_datetime(
                        [
                            "2021-01-01",
                            "2021-01-02",
                            "2021-01-03",
                            "2021-01-04",
                            "2021-01-05",
                        ]
                    ),
                    "corridor1": [200, 300, 400, 500, 400],
                    "corridor2": [1000, 3000, 4000, 5000, 4000],
                    "flag": [1, 0, 0, 0, 0],
                }
            ),
            pd.to_datetime("2021-01-03"),
            pd.DataFrame(
                {
                    "transaction_date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                    "corridor1": [200, 300],
                    "corridor2": [1000, 3000],
                    "flag": [1, 0],
                }
            ),
            pd.DataFrame(
                {
                    "transaction_date": pd.to_datetime(
                        ["2021-01-03", "2021-01-04", "2021-01-05"]
                    ),
                    "corridor1": [400, 500, 400],
                    "corridor2": [4000, 5000, 4000],
                    "flag": [0, 0, 0],
                }
            ),
            {"corridor1": 600, "corridor2": 6000, "flag": 2},
            {"corridor1": 99.0, "corridor2": 499.0, "flag": -1},
        ),
        (
            pd.DataFrame(
                {
                    "transaction_date": pd.to_datetime(
                        [
                            "2021-01-01",
                            "2021-01-02",
                            "2021-01-03",
                            "2021-01-04",
                            "2021-01-05",
                        ]
                    ),
                    "corridor1": [200, 300, 400, 500, 400],
                    "corridor2": [1000, 3000, 4000, 5000, 4000],
                    "flag": [1, 0, 0, 0, 0],
                }
            ),
            pd.to_datetime("2021-02-01"),
            pd.DataFrame(
                {
                    "transaction_date": pd.to_datetime(
                        [
                            "2021-01-01",
                            "2021-01-02",
                            "2021-01-03",
                            "2021-01-04",
                            "2021-01-05",
                        ]
                    ),
                    "corridor1": [200, 300, 400, 500, 400],
                    "corridor2": [1000, 3000, 4000, 5000, 4000],
                    "flag": [1, 0, 0, 0, 0],
                }
            ),
            pd.DataFrame(),
            {"corridor1": 1000, "corridor2": 10000, "flag": 2},
            {"corridor1": 99.0, "corridor2": 499.0, "flag": -1},
        ),
    ],
    ids=["train-end is within ts", "train-end in the future"],
)
def test_split_train_and_test(
    input_df, train_end, train_expected, test_expected, caps_expected, floors_expected
):
    df_train_actual, df_test_actual, caps_actual, floors_actual = split_train_and_test(
        input_df, train_end=train_end
    )

    assert_frame_equal(df_train_actual, train_expected)
    assert_frame_equal(df_test_actual, df_test_actual)
    assert caps_actual == caps_expected
    assert floors_actual == floors_expected


@pytest.mark.parametrize(
    "preds_start, preds_end, expected_df",
    [
        (
            pd.to_datetime("2021-01-01"),
            pd.to_datetime("2021-01-10"),
            pd.DataFrame(
                {
                    "ds": pd.to_datetime(
                        [
                            "2021-01-01",
                            "2021-01-02",
                            "2021-01-03",
                            "2021-01-04",
                            "2021-01-05",
                            "2021-01-06",
                            "2021-01-07",
                            "2021-01-08",
                            "2021-01-09",
                            "2021-01-10",
                        ]
                    ),
                    "last_working_day_of_month_flag": [0.0] * 10,
                    "first_working_day_of_month_flag": [1.0] + [0.0] * 9,
                    "new_year_day_flag": [1.0] + [0.0] * 9,
                    "last_friday_of_month_flag": [0.0] * 10,
                }
            ),
        )
    ],
    ids=["10 day forecast horizon"],
)
def test_get_empty_ts_for_predictions(preds_start, preds_end, expected_df):
    actual_df = get_empty_ts_for_predictions(preds_start, preds_end)
    assert_frame_equal(actual_df, expected_df)


def test_merge_correspondent_data():
    df_global = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                    "2021-01-05",
                ]
            ),
            "gross_transactions": [200, 300, 400, 500, 400],
        }
    )
    df_correspondents = pd.DataFrame(
        {
            TRANSACTION_DATE: pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                    "2021-01-05",
                    "2021-01-06",
                ]
            ),
            "correspondent_name": ["u", "v", "w", "x", "y", "z"],
            "currency_code": ["c1", "c1", "c2", "c2", "c3", "c3"],
            "gross_corr_amount_corr_curr": [100, 200, 300, 200, 50, 20],
            "gross_revenue_gbp": [1, 2, 5, 3, 2, 8],
        }
    )
    expected_df = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                    "2021-01-04",
                    "2021-01-05",
                ]
            ),
            "gross_transactions": [200, 300, 400, 500, 400],
            "u_c1": [
                100.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "v_c1": [
                0.0,
                200,
                0.0,
                0.0,
                0.0,
            ],
            "w_c2": [
                0.0,
                0.0,
                300,
                0.0,
                0.0,
            ],
            "x_c2": [
                0.0,
                0.0,
                0.0,
                200,
                0.0,
            ],
            "y_c3": [
                0.0,
                0.0,
                0.0,
                0.0,
                50,
            ],
            "z_c3": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        }
    )

    actual_df = merge_correspondent_data(df_global, df_correspondents)

    assert_frame_equal(actual_df, expected_df)


import pandas as pd
from pandas.testing import assert_frame_equal

from src.modelling.prophet_model import train_models


def test_train_models():
    train_df = pd.read_csv("data/test_train_data.csv")
    caps = {"corr1": 185000000, "corr2": 151000000, "corr3": 190000000}
    floors = {"corr1": 0, "corr2": 1500000, "corr3": 1780000}

    fitted_models = train_models(train_df, caps, floors, regressors=[])

    corr1_history = pd.read_csv("data/corr1_history.csv", parse_dates=["ds"])
    corr2_history = pd.read_csv("data/corr2_history.csv", parse_dates=["ds"])
    corr3_history = pd.read_csv("data/corr3_history.csv", parse_dates=["ds"])

    assert_frame_equal(fitted_models["corr1"].history, corr1_history)
    assert_frame_equal(fitted_models["corr2"].history, corr2_history)
    assert_frame_equal(fitted_models["corr3"].history, corr3_history)
