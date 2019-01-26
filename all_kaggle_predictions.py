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

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def predictForDate(date, super_merged_df, rf_class):
    x = super_merged_df.loc[merged_df['Date'] == date]
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

seed = 7
np.random.seed(seed)

result_str_lst = list()
super_merged_df = DataFrame()

dir_path = "C:/Users/Katja/Desktop/Big Data/Projekt/data/"
dir_path = normpath(dir_path)
print(dir_path)

# read tickes file names
ticker_path = join(dir_path, "stocks")
ticker_file_names = [f for f in listdir(ticker_path) if (isfile(join(ticker_path, f)) and ".csv" in f)]

for file_name in ticker_file_names:

    ticker_name = file_name.replace(".csv", "")

    # 1 Load data frame
    file_str = join(ticker_path, file_name)
    print(file_str)
    df_ticker = pd.read_csv(file_str)
    if df_ticker.shape[0] < 212:
        print(ticker_name + " empty")
        continue
    df_ticker.columns = ['Date', 'Open', 'Close', 'Low', 'High', 'Volume']

    # 2 Prepare data frame
    # 2a Price
    df_ticker['Mid_prices'] = (df_ticker.Low + df_ticker.High) / 2
    df_for_midprices = df_ticker.copy(deep=True)
    df_midprices = df_for_midprices.drop(
        ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)

    midprices = df_midprices.values
    reframed_midprices = series_to_supervised(midprices, 0, 91)

    entwicklungsrate_midprices_10 = (
        1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+10)'])-1
    entwicklungsrate_midprices_20 = (
        1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+20)'])-1
    entwicklungsrate_midprices_30 = (
        1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+30)'])-1
    entwicklungsrate_midprices_40 = (
        1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+40)'])-1
    entwicklungsrate_midprices_50 = (
        1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+50)'])-1
    entwicklungsrate_midprices_60 = (
        1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+60)'])-1
    entwicklungsrate_midprices_70 = (
        1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+70)'])-1
    entwicklungsrate_midprices_80 = (
        1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+80)'])-1
    entwicklungsrate_midprices_90 = (
        1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+90)'])-1

    df_for_midprices['Entwicklungsrate Preis t+10'] = entwicklungsrate_midprices_10
    df_for_midprices['Entwicklungsrate Preis t+20'] = entwicklungsrate_midprices_20
    df_for_midprices['Entwicklungsrate Preis t+30'] = entwicklungsrate_midprices_30
    df_for_midprices['Entwicklungsrate Preis t+40'] = entwicklungsrate_midprices_40
    df_for_midprices['Entwicklungsrate Preis t+50'] = entwicklungsrate_midprices_50
    df_for_midprices['Entwicklungsrate Preis t+60'] = entwicklungsrate_midprices_60
    df_for_midprices['Entwicklungsrate Preis t+70'] = entwicklungsrate_midprices_70
    df_for_midprices['Entwicklungsrate Preis t+80'] = entwicklungsrate_midprices_80
    df_for_midprices['Entwicklungsrate Preis t+90'] = entwicklungsrate_midprices_90
    df_for_midprices = df_for_midprices.drop(
        ['Open', 'High', 'Low', 'Close', 'Volume', 'Mid_prices'], axis=1)

    # 2b Volume
    df_for_volumes = df_ticker.copy(deep=True)
    df_volumes = df_for_volumes.drop(
        ['Date', 'Open', 'High', 'Low', 'Close', 'Mid_prices'], axis=1)

    volumes = df_volumes.values
    reframed_volumes = series_to_supervised(volumes, 0, 91)

    entwicklungsrate_volumes_10 = (
        1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+10)'])-1
    entwicklungsrate_volumes_20 = (
        1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+20)'])-1
    entwicklungsrate_volumes_30 = (
        1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+30)'])-1
    entwicklungsrate_volumes_40 = (
        1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+40)'])-1
    entwicklungsrate_volumes_50 = (
        1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+50)'])-1
    entwicklungsrate_volumes_60 = (
        1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+60)'])-1
    entwicklungsrate_volumes_70 = (
        1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+70)'])-1
    entwicklungsrate_volumes_80 = (
        1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+80)'])-1
    entwicklungsrate_volumes_90 = (
        1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+90)'])-1

    df_for_volumes['Entwicklungsrate Volume t+10'] = entwicklungsrate_volumes_10
    df_for_volumes['Entwicklungsrate Volume t+20'] = entwicklungsrate_volumes_20
    df_for_volumes['Entwicklungsrate Volume t+30'] = entwicklungsrate_volumes_30
    df_for_volumes['Entwicklungsrate Volume t+40'] = entwicklungsrate_volumes_40
    df_for_volumes['Entwicklungsrate Volume t+50'] = entwicklungsrate_volumes_50
    df_for_volumes['Entwicklungsrate Volume t+60'] = entwicklungsrate_volumes_60
    df_for_volumes['Entwicklungsrate Volume t+70'] = entwicklungsrate_volumes_70
    df_for_volumes['Entwicklungsrate Volume t+80'] = entwicklungsrate_volumes_80
    df_for_volumes['Entwicklungsrate Volume t+90'] = entwicklungsrate_volumes_90
    df_for_volumes = df_for_volumes.drop(
        ['Open', 'High', 'Low', 'Close', 'Volume', 'Mid_prices'], axis=1)

    # 3 Merge data frames
    merged_df = pd.merge(df_for_volumes, df_for_midprices, on='Date')
    merged_df.dropna(axis = 0, inplace = True)
    merged_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)
    merged_df = merged_df[~(merged_df['Entwicklungsrate Volume t+10']==np.inf)]

    # 4 Add dependend variable
    df_train_label = pd.read_csv(
        join(dir_path, 'labels_train.csv'), header=0, index_col=0)

    df_train_label = df_train_label.loc[:, df_train_label.columns.intersection([
        ticker_name])]
    df_train_label.columns = ['Y']

    df_complete = pd.merge(merged_df, df_train_label[['Y']], on='Date')
    df_complete = df_complete.sort_values('Date')
    
    super_merged_df = super_merged_df.append(df_complete, ignore_index=True)
    
count = super_merged_df.shape[0]
print(count)

print(super_merged_df.tail())

# 5 Splitting Data in training and testing
x = super_merged_df[['Entwicklungsrate Preis t+10',
                    'Entwicklungsrate Preis t+20',
                    'Entwicklungsrate Preis t+30',
                    'Entwicklungsrate Preis t+40',
                    'Entwicklungsrate Preis t+50',
                    'Entwicklungsrate Preis t+60',
                    'Entwicklungsrate Preis t+70',
                    'Entwicklungsrate Preis t+80',
                    'Entwicklungsrate Preis t+90',
                    'Entwicklungsrate Volume t+10',
                    'Entwicklungsrate Volume t+20',
                    'Entwicklungsrate Volume t+30',
                    'Entwicklungsrate Volume t+40',
                    'Entwicklungsrate Volume t+50',
                    'Entwicklungsrate Volume t+60',
                    'Entwicklungsrate Volume t+70',
                    'Entwicklungsrate Volume t+80',
                    'Entwicklungsrate Volume t+90']]
y = super_merged_df['Y']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42)
print(y_train.shape)

# 6 Random Forest
rf_class = RandomForestClassifier(n_estimators=100, random_state=42)

rf_class.fit(x_train, y_train)

rf_score = rf_class.score(x_test, y_test)
print(rf_score)




# 7 Prediction for Kaggle
selected_dates_df = DataFrame()
selected_dates = list()
default_dates = list()

for date in boersen_days_2018:
    x = super_merged_df.loc[super_merged_df['Date'] == date]
    if not x.empty:
        selected_dates_df = selected_dates_df.append(x)
        selected_dates.append(date)
    else:
        default_dates.append(date)

#selected_dates_df = selected_dates_df.drop(['Date'], axis=1)
selected_dates_df.head()

prediction = rf_class.predict(selected_dates_df)
# print(prediction)

for i in range(len(prediction) + len(default_dates)):
    if len(selected_dates) > 0:
        min_pred = min(selected_dates)
    else:
        min_pred = '9999-99-99'
    if len(default_dates) > 0:
        min_default = min(default_dates)
    else:
        min_default = '9999-99-99'

    if min_pred < min_default:
        min_pred_idx = selected_dates.index(min_pred)
        result_str_lst.append(
            [min_pred + ":" + ticker_name, prediction[min_pred_idx]])
        del selected_dates[min_pred_idx]
        prediction = np.delete(prediction, min_pred_idx)
    else:
        min_default_idx = default_dates.index(min_default)
        result_str_lst.append(
            [min_default + ":" + ticker_name, 0])
        del default_dates[min_default_idx]

# 8 Transfer list to DataFrame and save
kaggle = pd.DataFrame(data=result_str_lst, columns=['Id', 'Category'])
kaggle.shape

kaggle = kaggle.to_csv('kaggle_Rapp_Katja.csv', index=False)







# date_list = daterange('2018-01-02', '2018-06-30')

# selected_dates_df = DataFrame()
# selected_dates = list()

# for date in date_list:
#     x = merged_df.loc[merged_df['Date'] == date.strftime("%Y-%m-%d")]
#     if not x.empty:
#         selected_dates_df = selected_dates_df.append(x)
#         selected_dates.append(date.strftime("%Y-%m-%d"))

# selected_dates_df = selected_dates_df.drop(['Date'], axis=1)
# selected_dates_df.head()

# prediction = rf_class.predict(selected_dates_df)
# # print(prediction)

# for i in range(len(prediction)):
#     # print(selected_dates[i] + ": " + ticker_name + ", " + str(prediction[i]))
#     result_str_lst.append(
#         [selected_dates[i] + ":" + ticker_name, prediction[i]])

# # Transfer list to DataFrame and save
# kaggle = pd.DataFrame(data=result_str_lst, columns=['Id', 'Category'])
# kaggle.shape

# kaggle = kaggle.to_csv('kaggle_Rapp_Katja.csv', index=False)
