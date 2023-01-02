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