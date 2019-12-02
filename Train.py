# -*- coding: utf-8 -*-
"""
@author: cyx
"""

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score


# calculate error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# divide datset into train set and test set
def timeseries_split(x, y, test_size):
    index_test = int(len(x) * (1 - test_size))
    x_train = x.iloc[:index_test]
    y_train = y.iloc[:index_test]
    x_test = x.iloc[index_test:]
    y_test = y.iloc[index_test:]
    return x_train, y_train, x_test, y_test

# calculate mean
def cal_mean(data, x_feature, y_feature):
    return dict(data.groupby(x_feature)[y_feature].mean())

# build feature for datetime
def build_feature(data, lag_start, lag_end, test_size, target_encoding=False):
    df = data
    df.set_index("Datetime", drop=True, inplace=True)
    # make feature
    # shift feature，shift from lag_start to lag_end
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
    # data split
    y = df.dropna().y
    x = df.dropna().drop("y", axis=1)
    x_train, y_train, x_test, y_test = \
        timeseries_split(x, y, test_size=test_size)
    return x_train, y_train, x_test, y_test

# draw results
def plot_result(y, y_fit):
    assert len(y) == len(y_fit)
    plt.figure(figsize=(16, 8))
    plt.plot(y.index, y, label="y_orig")
    plt.plot(y.index, y_fit, label="y_fit")
    error = mean_absolute_error(y, y_fit)
    plt.title("mean_absolute_error{0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    dataf = pd.read_csv("./dataset/train dataset.csv")
    data = dataf
    dataf["Datetime"] = pd.to_datetime(dataf['Datetime'],format = '%Y-%m-%d %H:%M')
    dataf = dataf.drop(['Capacity'],axis=1)
    dataf = dataf.sort_values("Datetime")
    dataf.rename(columns={"Occupancy": "y"}, inplace=True)
    lag_start = 18 #shift feature
    lag_end = 19
    x_train, y_train, x_test, y_test = build_feature(dataf, lag_start=lag_start, lag_end=lag_end, test_size=0.3, target_encoding=True)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    tscv = TimeSeriesSplit(n_splits=5)

    lr = RandomForestRegressor(n_estimators=100, n_jobs = -1)

    X = np.concatenate((x_train_scaled, x_test_scaled))
    Y = pd.concat([y_train, y_test])

    print(cross_val_score(lr,X,Y,cv = 10).mean())
    lr.fit(x_train_scaled, y_train)
    # save model
    joblib.dump(lr, "trained_model.m")
 
    y = pd.concat([y_train, y_test])
    y_pred = lr.predict(x_test_scaled)
    # show results
    #plt.figure(figsize=(16, 8))
    #plt.plot(data["Datetime"], data["Occupancy"])
    plot_result(y_test, y_pred)