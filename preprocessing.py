from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meteostat import Daily, Hourly, Point
from pandas.api.types import is_string_dtype
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def merge_data(df: pd.DataFrame):
    start, end = (
        df["Timestamp"][df.index[0]],
        df.loc[:, "Timestamp"][df[df["Room"] == "Raum 004"].index[-1]],
    )
    start, end = datetime.strptime(start, "%d.%m.%Y %H:%M"), datetime.strptime(
        end, "%d.%m.%Y %H:%M"
    )
    phoenix = Point(51.98589, 9.25246, 70)
    # Get daily data for 2018
    data = Hourly(phoenix, start, end + timedelta(weeks=4))
    tdf = data.fetch()
    tdf = tdf[["dwpt", "rhum", "prcp", "wspd", "tsun", "coco"]]

    tdf = tdf.resample("15T", kind="timestamp").interpolate()

    print(tdf.shape, df.shape)

    # merge
    df.set_index("Timestamp", inplace=True)
    df.index = pd.to_datetime(df.index, format="%d.%m.%Y %H:%M")

    res = df.join(tdf, how="left").drop(columns=["WindVelocity", "RelativeHumidity"])

    return res


def preprocess_df(
    df: pd.DataFrame, method: str = "zero", seed: int = 69
) -> pd.DataFrame:
    df = convert_df_to_numeric(df)

    # impute missing numeric values
    if method == "zero":
        df.fillna(0.0, inplace=True)
    else:
        imp = IterativeImputer(
            missing_values=np.nan,
            add_indicator=True,
            random_state=seed,
            n_nearest_features=5,
            sample_posterior=True,
        )
        df = imp.fit_transform(df)

    # encode categorical values
    enc = OrdinalEncoder()
    enc.fit(df[["Room", "BucketAttendees"]])
    df[["Room", "BucketAttendees"]] = enc.transform(df[["Room", "BucketAttendees"]])

    # standartization values
    # scaler = StandardScaler()
    # df.iloc[:, 3:18] = scaler.fit_transform(df.iloc[:, 3:18])

    return df


def convert_df_to_numeric(df: pd.DataFrame):
    # iterate over float columns
    for column in df.iloc[:, 1:]:
        if is_string_dtype(df[column]):
            df[column] = df[column].str.replace(",", ".").astype(np.float32)

    return df
