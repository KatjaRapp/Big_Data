from os import listdir
from os.path import isfile, join, normpath
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
from sklearn import preprocessing
from pandas import DataFrame
from pandas import concat
# %matplotlib inline
import datetime
from datetime import timedelta, date
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV,  cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import csv

boersen_days_2018 = [
    "2018-01-02",
    "2018-01-03",
    "2018-01-04",
    "2018-01-05",
    "2018-01-08",
    "2018-01-09",
    "2018-01-10",
    "2018-01-11",
    "2018-01-12",
    "2018-01-16",
    "2018-01-17",
    "2018-01-18",
    "2018-01-19",
    "2018-01-22",
    "2018-01-23",
    "2018-01-24",
    "2018-01-25",
    "2018-01-26",
    "2018-01-29",
    "2018-01-30",
    "2018-01-31",
    "2018-02-01",
    "2018-02-02",
    "2018-02-05",
    "2018-02-06",
    "2018-02-07",
    "2018-02-08",
    "2018-02-09",
    "2018-02-12",
    "2018-02-13",
    "2018-02-14",
    "2018-02-15",
    "2018-02-16",
    "2018-02-20",
    "2018-02-21",
    "2018-02-22",
    "2018-02-23",
    "2018-02-26",
    "2018-02-27",
    "2018-02-28",
    "2018-03-01",
    "2018-03-02",
    "2018-03-05",
    "2018-03-06",
    "2018-03-07",
    "2018-03-08",
    "2018-03-09",
    "2018-03-12",
    "2018-03-13",
    "2018-03-14",
    "2018-03-15",
    "2018-03-16",
    "2018-03-19",
    "2018-03-20",
    "2018-03-21",
    "2018-03-22",
    "2018-03-23",
    "2018-03-26",
    "2018-03-27",
    "2018-03-28",
    "2018-03-29",
    "2018-04-02",
    "2018-04-03",
    "2018-04-04",
    "2018-04-05",
    "2018-04-06",
    "2018-04-09",
    "2018-04-10",
    "2018-04-11",
    "2018-04-12",
    "2018-04-13",
    "2018-04-16",
    "2018-04-17",
    "2018-04-18",
    "2018-04-19",
    "2018-04-20",
    "2018-04-23",
    "2018-04-24",
    "2018-04-25",
    "2018-04-26",
    "2018-04-27",
    "2018-04-30",
    "2018-05-01",
    "2018-05-02",
    "2018-05-03",
    "2018-05-04",
    "2018-05-07",
    "2018-05-08",
    "2018-05-09",
    "2018-05-10",
    "2018-05-11",
    "2018-05-14",
    "2018-05-15",
    "2018-05-16",
    "2018-05-17",
    "2018-05-18",
    "2018-05-21",
    "2018-05-22",
    "2018-05-23",
    "2018-05-24",
    "2018-05-25",
    "2018-05-29",
    "2018-05-30",
    "2018-05-31",
    "2018-06-01",
    "2018-06-04",
    "2018-06-05",
    "2018-06-06",
    "2018-06-07",
    "2018-06-08",
    "2018-06-11",
    "2018-06-12",
    "2018-06-13",
    "2018-06-14",
    "2018-06-15",
    "2018-06-18",
    "2018-06-19",
    "2018-06-20",
    "2018-06-21",
    "2018-06-22",
    "2018-06-25",
    "2018-06-26",
    "2018-06-27",
    "2018-06-28",
    "2018-06-29"
]

boersen_days_2017 = [
    "2017-08-23",
    "2017-08-24",
    "2017-08-25",
    "2017-08-28",
    "2017-08-29",
    "2017-08-30",
    "2017-08-31",
    "2017-09-01",
    "2017-09-05",
    "2017-09-06",
    "2017-09-07",
    "2017-09-08",
    "2017-09-11",
    "2017-09-12",
    "2017-09-13",
    "2017-09-14",
    "2017-09-15",
    "2017-09-18",
    "2017-09-19",
    "2017-09-20",
    "2017-09-21",
    "2017-09-22",
    "2017-09-25",
    "2017-09-26",
    "2017-09-27",
    "2017-09-28",
    "2017-09-29",
    "2017-10-02",
    "2017-10-03",
    "2017-10-04",
    "2017-10-05",
    "2017-10-06",
    "2017-10-09",
    "2017-10-10",
    "2017-10-11",
    "2017-10-12",
    "2017-10-13",
    "2017-10-16",
    "2017-10-17",
    "2017-10-18",
    "2017-10-19",
    "2017-10-20",
    "2017-10-23",
    "2017-10-24",
    "2017-10-25",
    "2017-10-26",
    "2017-10-27",
    "2017-10-30",
    "2017-10-31",
    "2017-11-01",
    "2017-11-02",
    "2017-11-03",
    "2017-11-06",
    "2017-11-07",
    "2017-11-08",
    "2017-11-09",
    "2017-11-10",
    "2017-11-13",
    "2017-11-14",
    "2017-11-15",
    "2017-11-16",
    "2017-11-17",
    "2017-11-20",
    "2017-11-21",
    "2017-11-22",
    "2017-11-24",
    "2017-11-27",
    "2017-11-28",
    "2017-11-29",
    "2017-11-30",
    "2017-12-01",
    "2017-12-04",
    "2017-12-05",
    "2017-12-06",
    "2017-12-07",
    "2017-12-08",
    "2017-12-11",
    "2017-12-12",
    "2017-12-13",
    "2017-12-14",
    "2017-12-15",
    "2017-12-18",
    "2017-12-19",
    "2017-12-20",
    "2017-12-21",
    "2017-12-22",
    "2017-12-26",
    "2017-12-27",
    "2017-12-28",
    "2017-12-29"
]

all_boersen_days = boersen_days_2017
all_boersen_days.extend(boersen_days_2018)

features_time_range = [0]#[i*10 for i in range(11)]
features_used = ['Open', 'Close', 'Low', 'High'] # 'Volume'

def series_to_supervised(feature_name, data, time_range=[0], dropnan=True):
    df = DataFrame(data)
    cols, names = list(), list()
    # forecast sequence (t, t+1, ... t+n)
    for i in time_range:
        cols.append(df.shift(-i))
        names += [feature_name + "_" + str(i)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def predictForDate(date, merged_df, rf_class):
    x = merged_df.loc[merged_df['Date'] == date]
    x = x.drop(['Date'], axis=1)
    prediction = rf_class.predict(x)
    return prediction


def daterange(start_date_str, end_date_str):
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')

    dates = list()
    for n in range(int((end_date - start_date).days)):
        dates.append(start_date + timedelta(n))
    return dates

def main():

    seed = 7
    np.random.seed(seed)

    result_str_lst = list()

    dir_path = "/home/ubuntu/data"
    dir_path = normpath(dir_path)

    # load ticker file names
    ticker_path = join(dir_path, "stocks")
    ticker_file_names = [f for f in listdir(ticker_path) if (isfile(join(ticker_path, f)) and ".csv" in f)]

    # load training labels
    df_train_label = pd.read_csv(join(dir_path, 'labels_train.csv'), header=0, index_col=0)

    all_stocks_df = DataFrame()
    first_stock_b = True

    for file_name in ticker_file_names:

        ticker_name = file_name.replace(".csv", "")

        # 1 Load data frame
        file_str = join(ticker_path, file_name)
        df_ticker = pd.read_csv(file_str)
        df_ticker.columns = ['Date', 'Open', 'Close', 'Low', 'High', 'Volume']

        ticker_data_valid = True
        for date_str in all_boersen_days:
            if not date_str in df_ticker['Date'].values:
                ticker_data_valid = False
                continue

        # if ticker is not valid, then write all zeros in the output file
        if not ticker_data_valid:
            for date_str in boersen_days_2018:
                result_str_lst.append([date_str + ":" + ticker_name, 0])
            print(ticker_name + " not complete")
            continue


        stock_merged_df = DataFrame()
        first_feature_b = True

        for feature in features_used:
            feature_ = feature + "_"

            feature_df = df_ticker.copy(deep=True)
            # drop everything except of the current feature and the date
            feature_df = feature_df[["Date", feature]]

            timeseries_feature_df = series_to_supervised(feature, feature_df[feature].values, features_time_range)

            for t in features_time_range:
                if not t == features_time_range[0]:
                    f_0 = timeseries_feature_df[feature_ + str(features_time_range[0])]
                    f_t = timeseries_feature_df[feature_ + str(t)]
                    f_t = (f_t - f_0) / f_0
                    feature_df[feature_ + str(t)] = f_t
            feature_df = feature_df.drop([feature], axis=1)

            feature_df = feature_df.dropna(axis = 0)
            feature_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)

            if first_feature_b:
                stock_merged_df = feature_df
                first_feature_b = False
            else:
                stock_merged_df = pd.merge(stock_merged_df, feature_df, on='Date')

        stock_merged_df = stock_merged_df.dropna(axis = 0)
        stock_merged_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)

        # 4 Add dependend variable
        labels_df = df_train_label.loc[:, df_train_label.columns.intersection([ticker_name])]
        labels_df = labels_df.rename(index=str, columns={ticker_name: 'Y'})
        stock_complete_df = pd.merge(stock_merged_df, labels_df[['Y']], on='Date')
        stock_complete_df = stock_complete_df.sort_values('Date')

        if first_stock_b:
            all_stocks_df = stock_complete_df
            first_stock_b = False
        else:
            all_stocks_df = all_stocks_df.append(stock_complete_df, ignore_index=True)
            all_stocks_df = all_stocks_df.sort_values('Date')


    all_stocks_df = all_stocks_df.dropna(axis = 0)
    all_stocks_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)

    column_names = list(all_stocks_df)
    column_names.remove("Date")
    column_names.remove("Y")
    print(column_names)


    # column_names = ['Open_10', 'Open_20', 'Open_30', 'Open_40', 'Open_50', 'Open_60', 'Open_70', 'Open_80', 'Open_90', 'Open_100',
    # 'Close_10', 'Close_20', 'Close_30', 'Close_40', 'Close_50', 'Close_60', 'Close_70', 'Close_80', 'Close_90',
    # 'Close_100', 'Low_10', 'Low_20', 'Low_30', 'Low_40', 'Low_50', 'Low_60', 'Low_70', 'Low_80', 'Low_90', 'Low_100',
    # 'High_10', 'High_20', 'High_30', 'High_40', 'High_50', 'High_60', 'High_70', 'High_80', 'High_90', 'High_100',
    # 'Volume_20', 'Volume_30', 'Volume_40', 'Volume_50']#, 'Volume_60', 'Volume_70', 'Volume_80', 'Volume_90',
    # # 'Volume_100']

    # # 'Volume_10'

    x = all_stocks_df[column_names]

    print(x.head())

    y = all_stocks_df["Y"]
    print(y.head())

    # all_stocks_df = all_stocks_df.to_csv('kaggle_Rapp_Katja.csv', index=False)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=42)

    # 6 Random Forest
    rf_class = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_class.fit(x_train, y_train)

    rf_score = rf_class.score(x_test, y_test)
    print("Random Forest Score: " + ticker_name + ": " + str(rf_score))

    #     # 7 Prediction for Kaggle
    #     # prediction = rf_class.predict(x_test)
    #     # prediction

    #     # merged_df.loc[merged_df['Date'] == '2018-01-02']

    #     # x = merged_df.loc[merged_df['Date'] == '2018-06-11']
    #     # x = x.drop(['Date'], axis=1)
    #     # prediction = rf_class.predict(x)
    #     # prediction

    #     # res = predictForDate('2018-06-11', merged_df, rf_class)
    #     # print(res)

    #     # date_list = daterange('2018-01-02', '2018-06-30')

    #     selected_dates_df = DataFrame()
    #     selected_dates = list()
    #     default_dates = list()

    #     for date in boersen_days_2018:
    #         x = merged_df.loc[merged_df['Date'] == date]
    #         if not x.empty:
    #             selected_dates_df = selected_dates_df.append(x)
    #             selected_dates.append(date)
    #         else:
    #             default_dates.append(date)


    #     selected_dates_df = selected_dates_df.drop(['Date'], axis=1)
    #     selected_dates_df.head()

    #     prediction = rf_class.predict(selected_dates_df)

    #     for i in range(len(prediction) + len(default_dates)):
    #         if len(selected_dates) > 0:
    #             min_pred = min(selected_dates)
    #         else:
    #             min_pred = '9999-99-99'
    #         if len(default_dates) > 0:
    #             min_default = min(default_dates)
    #         else:
    #             min_default = '9999-99-99'

    #         if min_pred < min_default:
    #             min_pred_idx = selected_dates.index(min_pred)
    #             result_str_lst.append(
    #                 [min_pred + ":" + ticker_name, prediction[min_pred_idx]])
    #             del selected_dates[min_pred_idx]
    #             prediction = np.delete(prediction, min_pred_idx)
    #         else:
    #             min_default_idx = default_dates.index(min_default)
    #             result_str_lst.append(
    #                 [min_default + ":" + ticker_name, 0])
    #             del default_dates[min_default_idx]



    # # Transfer list to DataFrame and save
    # kaggle = pd.DataFrame(data=result_str_lst, columns=['Id', 'Category'])
    # kaggle.shape

    # kaggle = kaggle.to_csv('kaggle_Rapp_Katja.csv', index=False)

if __name__ == "__main__":
    main()