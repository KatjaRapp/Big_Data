import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
from sklearn import preprocessing
from  pandas  import  DataFrame 
from  pandas  import  concat 
%matplotlib inline
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

# 1 Load data frame
seed = 7
np.random.seed(seed)

df_ticker = pd.read_csv("data/stocks/AAPL.csv")
df_ticker.columns = ['Date','Open','Close','Low','High','Volume']
df_ticker.head()

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

# 2 Prepare data frame
# 2a Price
df_ticker['Mid_prices'] = (df_ticker.Low + df_ticker.High) / 2
df_for_midprices = df_ticker.copy(deep=True)
df_midprices = df_for_midprices.drop(['Date','Open','High','Low','Close','Volume'], axis=1)
df_midprices.head()

midprices = df_midprices.values
reframed_midprices = series_to_supervised(midprices, 0, 91)
reframed_midprices.head()

entwicklungsrate_midprices_10 = (1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+10)'])-1
entwicklungsrate_midprices_20 = (1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+20)'])-1
entwicklungsrate_midprices_30 = (1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+30)'])-1
entwicklungsrate_midprices_40 = (1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+40)'])-1
entwicklungsrate_midprices_50 = (1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+50)'])-1
entwicklungsrate_midprices_60 = (1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+60)'])-1
entwicklungsrate_midprices_70 = (1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+70)'])-1
entwicklungsrate_midprices_80 = (1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+80)'])-1
entwicklungsrate_midprices_90 = (1/reframed_midprices['var1(t)']*reframed_midprices['var1(t+90)'])-1

df_for_midprices['Entwicklungsrate Preis t+10'] = entwicklungsrate_midprices_10
df_for_midprices['Entwicklungsrate Preis t+20'] = entwicklungsrate_midprices_20
df_for_midprices['Entwicklungsrate Preis t+30'] = entwicklungsrate_midprices_30
df_for_midprices['Entwicklungsrate Preis t+40'] = entwicklungsrate_midprices_40
df_for_midprices['Entwicklungsrate Preis t+50'] = entwicklungsrate_midprices_50
df_for_midprices['Entwicklungsrate Preis t+60'] = entwicklungsrate_midprices_60
df_for_midprices['Entwicklungsrate Preis t+70'] = entwicklungsrate_midprices_70
df_for_midprices['Entwicklungsrate Preis t+80'] = entwicklungsrate_midprices_80
df_for_midprices['Entwicklungsrate Preis t+90'] = entwicklungsrate_midprices_90
df_for_midprices = df_for_midprices.drop(['Open','High','Low','Close','Volume','Mid_prices'], axis=1)
df_for_midprices.head()

# 2b Volume
df_for_volumes = df_ticker.copy(deep=True)
df_volumes = df_for_volumes.drop(['Date','Open','High','Low','Close','Mid_prices'], axis=1)
df_volumes.head()

volumes = df_volumes.values
reframed_volumes = series_to_supervised(volumes, 0, 91)
reframed_volumes.head()

entwicklungsrate_volumes_10 = (1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+10)'])-1
entwicklungsrate_volumes_20 = (1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+20)'])-1
entwicklungsrate_volumes_30 = (1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+30)'])-1
entwicklungsrate_volumes_40 = (1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+40)'])-1
entwicklungsrate_volumes_50 = (1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+50)'])-1
entwicklungsrate_volumes_60 = (1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+60)'])-1
entwicklungsrate_volumes_70 = (1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+70)'])-1
entwicklungsrate_volumes_80 = (1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+80)'])-1
entwicklungsrate_volumes_90 = (1/reframed_volumes['var1(t)']*reframed_volumes['var1(t+90)'])-1

df_for_volumes['Entwicklungsrate Volume t+10'] = entwicklungsrate_volumes_10
df_for_volumes['Entwicklungsrate Volume t+20'] = entwicklungsrate_volumes_20
df_for_volumes['Entwicklungsrate Volume t+30'] = entwicklungsrate_volumes_30
df_for_volumes['Entwicklungsrate Volume t+40'] = entwicklungsrate_volumes_40
df_for_volumes['Entwicklungsrate Volume t+50'] = entwicklungsrate_volumes_50
df_for_volumes['Entwicklungsrate Volume t+60'] = entwicklungsrate_volumes_60
df_for_volumes['Entwicklungsrate Volume t+70'] = entwicklungsrate_volumes_70
df_for_volumes['Entwicklungsrate Volume t+80'] = entwicklungsrate_volumes_80
df_for_volumes['Entwicklungsrate Volume t+90'] = entwicklungsrate_volumes_90
df_for_volumes = df_for_volumes.drop(['Open','High','Low','Close','Volume','Mid_prices'], axis=1)
df_for_volumes.head()

# 3 Merge data frames
merged_df = pd.merge(df_for_volumes,df_for_midprices,on='Date')
merged_df.tail()

# 4 Add dependend variable
df_train_label = pd.read_csv('data/labels_train.csv', header=0, index_col=0)
df_train_label.head()

df_train_label = df_train_label.loc[:, df_train_label.columns.intersection(['AAPL'])]
df_train_label.columns = ['Y']
df_train_label.tail()

df_complete = pd.merge(merged_df,df_train_label[['Y']],on='Date')
df_complete = df_complete.sort_values('Date')
#df_complete = df_complete.drop('Date', axis=1)
df_complete.tail()

# 5 Splitting Data in training and testing
x = df_complete[['Entwicklungsrate Preis t+10', 
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
y = df_complete['Y']

x.shape, y.shape

x.max()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

# 6 Random Forest
rf_class = RandomForestClassifier(n_estimators=100, random_state=42)

rf_class.fit(x_train, y_train)

rf_class.score(x_test, y_test)

type(x_test)

# 7 Prediction for Kaggle
prediction = rf_class.predict(x_test)
prediction

merged_df.loc[merged_df['Date'] == '2018-01-02']

x = merged_df.loc[merged_df['Date'] == '2018-06-11']
x = x.drop(['Date'], axis=1)
prediction = rf_class.predict(x)
prediction

def predictForDate(date, merged_df):
    x = merged_df.loc[merged_df['Date'] == date]
    x = x.drop(['Date'], axis=1)
    prediction = rf_class.predict(x)
    return prediction

res = predictForDate('2018-06-11', merged_df)
print(res)

def daterange(start_date_str, end_date_str):
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    
    dates = list()
    for n in range(int ((end_date - start_date).days)):
        dates.append(start_date + timedelta(n))
    return dates

date_list = daterange('2018-01-02','2018-06-30')

selected_dates_df = DataFrame()
selected_dates = list()

for date in date_list:
    x = merged_df.loc[merged_df['Date'] == date.strftime("%Y-%m-%d")]
    if not x.empty:
        selected_dates_df = selected_dates_df.append(x)
        selected_dates.append(date.strftime("%Y-%m-%d"))

selected_dates_df = selected_dates_df.drop(['Date'], axis=1)
selected_dates_df.head()

prediction = rf_class.predict(selected_dates_df)
print(prediction)

result_str_lst = list()
for i in range(len(prediction)):
    print(selected_dates[i] + ": AAPL, " + str(prediction[i]))
    result_str_lst.append(selected_dates[i] + ": AAPL, " + str(prediction[i]))

# Transfer list to DataFrame and save
kaggle = pd.DataFrame(data=result_str_lst, columns = ['Id,Category'])
kaggle.shape

kaggle.to_csv('kaggle_Rapp_Katja.csv', index=False)