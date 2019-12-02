# -*- coding: utf-8 -*-
"""
@author: cyx
"""
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# calculate mean
def cal_mean(data, x_feature, y_feature):
    return dict(data.groupby(x_feature)[y_feature].mean())

# build feature for datetime
def build_feature (data, lag_start, lag_end, target_encoding=False, num_time_pred = 1):
    last_date = data["Datetime"].max()
    pred_points = int(num_time_pred *8) # prediction time
    pred_date = pd.date_range(start=last_date, periods=pred_points + 1, freq="30min")
    pred_date = pred_date[pred_date > last_date]  # exclude last_date, last_date is not predicted point
    future_data = pd.DataFrame({"Datetime": pred_date, "y": np.zeros(len(pred_date))})#set value in predicted time to 0，then build features
    # concat future data and last data
    df = pd.concat([data, future_data])
    df.set_index("Datetime", drop=True, inplace=True)
    # make feature
    # shift feature, shift from lag_start to lag_end
    for i in range(lag_start, lag_end):
        df["lag_{}".format(i)] = df.y.shift(i)
    #diff shift feature
    df["diff_lag_{}".format(lag_start)] = df["lag_{}".format(lag_start)].diff(1)
    # time feature
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["weekday"] = df.index.weekday
    df["weekend"] = df.weekday.isin([5, 6]) * 1
    # time mean feature replace time feature
    if target_encoding:
        df["weekday_avg"] = list(map(cal_mean(df, "weekday", "y").get, df.weekday))
        df["hour_avg"] = list(map(cal_mean(df, "hour", "y").get, df.hour))
        df["weekend_avg"] = list(map(cal_mean(df, "weekend", "y").get, df.weekend))
        df["minute_avg"] = list(map(cal_mean(df, "minute", "y").get, df.minute))
        df = df.drop(["hour","minute","weekday", "weekend"], axis = 1)
    x = df.dropna().drop("y", axis=1)
    x = x.tail(pred_points)
    return x


def predict_future():
    dataset = pd.read_csv("./dataset/parked number.csv")
    dataset["Datetime"] = pd.to_datetime(dataset['Datetime'],format = '%Y-%m-%d %H:%M')
    dataset = dataset.sort_values("Datetime")
    dataset.rename(columns= {"Occupancy": "y"}, inplace=True)
    lag_start = 18 # shift
    lag_end = 19
    lr = joblib.load("trained_model.m")
    x = build_feature(dataset, lag_start = lag_start, lag_end = lag_end, target_encoding=True, num_time_pred = 1)
    y_pred = lr.predict(x)
    return y_pred