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

#ticker names
# tickers_all = ['A', 'AAN', 'AAP', 'AAPL', 'AAXN', 'ABC', 'ABCD', 'ABG', 'ABM', 'ABMC',
#                'ABMD', 'ABT', 'ABTL', 'ACAD', 'ACAT', 'ACCO', 'ACET', 'ACGL', 'ACLS',
#                'ACN', 'ACOR', 'ACRX', 'ACW', 'ACY', 'ADBE', 'ADI', 'ADP', 'ADS', 'AEHR',
#                'AEIS', 'AEO', 'AES', 'AGCO', 'AGN', 'AGNC', 'AGX', 'AHPI', 'AHS', 'AIR',
#                'AIRI', 'AIT', 'AJRD', 'AKAM', 'AKRX', 'AKS', 'AL', 'ALB', 'ALE', 'ALG',
#                'ALGN', 'ALJ', 'ALK', 'ALNY', 'ALTR', 'ALV', 'ALX', 'ALXN', 'AMCX', 'AMD',
#                'AME', 'AMG', 'AMGN', 'AMKR', 'AMOT', 'AMP', 'AMT', 'AMTD', 'AMZN', 'AN',
#                'ANDV', 'ANSS', 'AON', 'AOS', 'AOSL', 'APA', 'APC', 'APD', 'APH', 'APOG',
#                'ARAY', 'ARCI', 'ARE', 'ARNC', 'ARRY', 'ARW', 'ASCMA', 'ASNA', 'ASTC',
#                'ATHN', 'ATI', 'ATO', 'ATR', 'ATRO', 'ATU', 'ATVI', 'ATW', 'AVA', 'AVB',
#                'AVGO', 'AVNW', 'AVP', 'AVY', 'AWI', 'AWK', 'AWRE', 'AYI', 'AZPN', 'AZZ',
#                'BA', 'BABY', 'BAX', 'BBBY', 'BBGI', 'BBRG', 'BC', 'BCO', 'BCPC', 'BCR',
#                'BDX', 'BG', 'BGG', 'BHE', 'BID', 'BIIB', 'BIO', 'BJRI', 'BKE', 'BLKB',
#                'BLL', 'BMRN', 'BMS', 'BMY', 'BONT', 'BRCD', 'BREW', 'BRO', 'BRS', 'BSPM',
#                'BSQR', 'BSX', 'BWLA', 'BX', 'BXP', 'BZH', 'C208', 'C730', 'CA', 'CAG',
#                'CAGU', 'CAH', 'CAL', 'CALD', 'CAMP', 'CAT', 'CAVM', 'CBB', 'CBG', 'CBI',
#                'CBK', 'CBM', 'CBOE', 'CBRL', 'CCF', 'CCI', 'CCK', 'CCL', 'CDE', 'CE',
#                'CELG', 'CERN', 'CF', 'CGA', 'CGNX', 'CHD', 'CHE', 'CHEF', 'CHK', 'CHRW',
#                'CHSP', 'CHTR', 'CIDM', 'CKP', 'CL', 'CLAR', 'CLC', 'CLCT', 'CLF', 'CLNE',
#                'CLR', 'CLRB', 'CLX', 'CMCSA', 'CMD', 'CME', 'CMG', 'CMI', 'CMLS', 'CMP',
#                'CNC', 'CNP', 'CNVR', 'COG', 'COH', 'COHR', 'COKE', 'COL', 'COLM', 'COP',
#                'COR', 'CORT', 'CPN', 'CPRT', 'CPSI', 'CPT', 'CR', 'CREE', 'CRIS', 'CRL',
#                'CRM', 'CRZO', 'CSCO', 'CSGP', 'CSGS', 'CSL', 'CSS', 'CSU', 'CSX', 'CTAS',
#                'CTL', 'CTSH', 'CTXS', 'CUB', 'CUBE', 'CUZ', 'CVS', 'CVX', 'CXO', 'CY',
#                'D', 'DAL', 'DCI', 'DDE', 'DE', 'DEST', 'DG', 'DGX', 'DHR', 'DIGIRAD',
#                'DIN', 'DISCA', 'DKS', 'DLB', 'DLTR', 'DLX', 'DNB', 'DNKN', 'DOV', 'DOW',
#                'DPS', 'DRI', 'DRQ', 'DSW', 'DTE', 'DVA', 'DVAX', 'DVN', 'DYNT', 'EBAY',
#                'EBIX', 'ECYT', 'ED', 'EEI', 'EFX', 'EGAN', 'EGHT', 'EIX', 'EL', 'ELLI',
#                'ELY', 'EME', 'EMN', 'EMR', 'ENDP', 'ENZ', 'EOG', 'EPD', 'EQIX', 'EQR',
#                'EQT', 'ES', 'ESL', 'ESP', 'ESRX', 'ESS', 'ETM', 'ETR', 'EV', 'EW', 'EXA',
#                'EXAS', 'EXPD', 'EXPE', 'EXPR', 'EXR', 'EXTR', 'F', 'FARO', 'FAST', 'FB',
#                'FBHS', 'FC', 'FCEL', 'FCS', 'FDS', 'FDX', 'FELE', 'FET', 'FFIV', 'FICO',
#                'FIS', 'FISV', 'FKWL', 'FL', 'FLIR', 'FLO', 'FLR', 'FLWS', 'FLXS', 'FOE',
#                'FONR', 'FORD', 'FORR', 'FOX', 'FRAN', 'FRD', 'FRED', 'FRGI', 'FRT',
#                'FSLR', 'FTEK', 'FTI', 'FTNT', 'FTR', 'FUL', 'G', 'GB', 'GBX', 'GCO',
#                'GD', 'GE', 'GENC', 'GEO', 'GEOS', 'GIII', 'GILD', 'GIS', 'GLW', 'GM',
#                'GME', 'GMED', 'GPC', 'GPRE', 'GPS', 'GRPN', 'GT', 'GUID', 'GWR', 'GWRE',
#                'GWW', 'GXP', 'H', 'HAL', 'HALO', 'HAS', 'HAYN', 'HBI', 'HCA', 'HCOM',
#                'HCP', 'HDSN', 'HEI', 'HES', 'HI', 'HLF', 'HLS', 'HNI', 'HNNA', 'HOG',
#                'HOLX', 'HON', 'HP', 'HPJ', 'HPQ', 'HRC', 'HRL', 'HSIC', 'HSII', 'HST',
#                'HSTM', 'HSY', 'HUN', 'HURN', 'HXL', 'IAIC', 'IBM', 'ICUI', 'IDA', 'IDCC',
#                'IFF', 'IGT', 'IHRT', 'IHT', 'IIN', 'ILMN', 'IMGN', 'IMO', 'IMPV', 'INCY',
#                'INFN', 'INT', 'INTC', 'INTU', 'IP', 'IPG', 'IPGP', 'IPI', 'IPXL', 'IRBT',
#                'ISIL', 'ISRG', 'ISSC', 'IT', 'ITRI', 'ITW', 'IVAC', 'IVDN', 'IVZ',
#                'JACK', 'JAZZ', 'JCI', 'JCOM', 'JCP', 'JKHY', 'JMBA', 'JNJ', 'JNPR',
#                'JOE', 'JOY', 'JSDA', 'JVA', 'JWN', 'KAR', 'KEM', 'KIRK', 'KLAC', 'KLIC',
#                'KMB', 'KMG', 'KMX', 'KND', 'KO', 'KOOL', 'KRO', 'KSU', 'KTEC', 'KWR',
#                'LAMR', 'LANC', 'LB', 'LCI', 'LCUT', 'LDOS', 'LEA', 'LEDS', 'LEG', 'LFGR',
#                'LG', 'LGL', 'LHCG', 'LII', 'LKQ', 'LLL', 'LMT', 'LNG', 'LNN', 'LNT',
#                'LOGI', 'LOPE', 'LOW', 'LPTH', 'LRCX', 'LTRX', 'LUK', 'LULU', 'LUV',
#                'LXRX', 'LYV', 'M', 'MA', 'MAC', 'MAMS', 'MAN', 'MANH', 'MAS', 'MAT',
#                'MCD', 'MCO', 'MDLZ', 'MDP', 'MDRX', 'MDSO', 'MDXG', 'MEIP', 'MHH', 'MHK',
#                'MIC', 'MITK', 'MJN', 'MKC', 'MKSI', 'MKTX', 'MLAB', 'MLM', 'MMM', 'MMS',
#                'MO', 'MON', 'MOV', 'MPC', 'MPWR', 'MRC', 'MRCY', 'MRK', 'MRO', 'MRVL',
#                'MSCC', 'MSCI', 'MSFT', 'MSI', 'MTN', 'MTSI', 'MTZ', 'MU', 'MUR', 'MXIM',
#                'MYL', 'NAII', 'NANO', 'NATI', 'NATR', 'NBIX', 'NCMI', 'NDSN', 'NE',
#                'NEE', 'NEM', 'NEPH', 'NFG', 'NFLX', 'NHTC', 'NI', 'NJR', 'NKE', 'NKTR',
#                'NLSNNV', 'NOC', 'NOV', 'NP', 'NPTN', 'NSSC', 'NTGR', 'NTRI', 'NUAN',
#                'NUE', 'NUTR', 'NUVA', 'NVDA', 'NVEC', 'NWL', 'NWY', 'NXST', 'NYT', 'O',
#                'OCC', 'OHI', 'OI', 'OKE', 'OLIN', 'OMC', 'OMCL', 'OMI', 'ON', 'ORCL',
#                'ORLY', 'OSIS', 'OSTK', 'OXM', 'OXY', 'PACB', 'PANW', 'PATK', 'PAYX',
#                'PBH', 'PBI', 'PBSV', 'PBYI', 'PCG', 'PCLN', 'PDFS', 'PEGA', 'PEP', 'PH',
#                'PII', 'PKG', 'PLAB', 'PLCE', 'PLD', 'PLPC', 'PM', 'PMD', 'PNR', 'PNRA',
#                'PNW', 'POOL', 'POST', 'PPG', 'PPL', 'PRGO', 'PRGS', 'PRLB', 'PRXL',
#                'PSA', 'PSDV', 'PSEG', 'PSTI', 'PSX', 'PTEN', 'PTN', 'PURE', 'PWR', 'PX',
#                'PXD', 'PZZA', 'QCOM', 'QEP', 'QNST', 'QSII', 'RAI', 'RAIL', 'RATE',
#                'RAVN', 'RAX', 'RCII', 'RCL', 'RDC', 'RDNT', 'RECN', 'REGN', 'REN', 'RFP',
#                'RGC', 'RGS', 'RHI', 'RHT', 'RICK', 'RIG', 'RLGY', 'RMCF', 'RMD', 'ROCK',
#                'ROK', 'ROL', 'ROP', 'ROST', 'RP', 'RPM', 'RS', 'RSG', 'RSYS', 'RT',
#                'RTN', 'RWC', 'RYN', 'SANM', 'SANW', 'SBAC', 'SBGI', 'SBUX', 'SCSC',
#                'SCX', 'SE', 'SEE', 'SEV', 'SGEN', 'SHLM', 'SHLO', 'SHO', 'SHOR', 'SHPGF',
#                'SHW', 'SIGM', 'SINO', 'SIRI', 'SJI', 'SKX', 'SLB', 'SLCA', 'SLM', 'SMCI',
#                'SMED', 'SMG', 'SMP', 'SMRT', 'SMSI', 'SMTC', 'SNA', 'SNAK', 'SNDK',
#                'SNHY', 'SNI', 'SNPS', 'SNX', 'SOFO', 'SOHO', 'SOHU', 'SONC', 'SORL',
#                'SPAN', 'SPG', 'SPLS', 'SPNC', 'SPPI', 'SPRT', 'SRE', 'SRT', 'SSI',
#                'SSNT', 'STMP', 'STRT', 'STZ', 'SVT', 'SWKS', 'SWN', 'SWX', 'SXT', 'SYK',
#                'SYNA', 'SYNT', 'T', 'TA', 'TAP', 'TBTC', 'TEL', 'TEN', 'TESO', 'TESS',
#                'TFX', 'TGI', 'TGLO', 'TGNA', 'TGT', 'THO', 'THS', 'TIF', 'TIS', 'TJX',
#                'TMO', 'TOL', 'TOWR', 'TRGP', 'TRIP', 'TRMB', 'TRN', 'TRNS', 'TROW',
#                'TRT', 'TSCO', 'TSN', 'TSRO', 'TTC', 'TTMI', 'TTWO', 'TUES', 'TUP',
#                'TWIN', 'TWX', 'TXMD', 'TYPE', 'TZOO', 'UAL', 'UG', 'UHAL', 'UHS', 'ULTA',
#                'UNFI', 'UNH', 'UNP', 'UPS', 'URBN', 'URI', 'USAP', 'USG', 'USNA', 'UTHR',
#                'UTMD', 'V', 'VAL', 'VAR', 'VDSI', 'VERU', 'VFC', 'VIAB', 'VIRC', 'VLGEA',
#                'VLO', 'VMC', 'VMI', 'VMW', 'VNO', 'VNTV', 'VRSK', 'VRSN', 'VSTM', 'VZ',
#                'WBC', 'WCG', 'WDC', 'WEC', 'WEN', 'WFM', 'WFT', 'WHR', 'WIFI', 'WIN',
#                'WINA', 'WLB', 'WLK', 'WM', 'WMAR', 'WMB', 'WMT', 'WNC', 'WOOF', 'WOR',
#                'WPX', 'WR', 'WRI', 'WSCI', 'WSM', 'WSTL', 'WTR', 'WU', 'WWE', 'WYN',
#                'WYNN', 'XCO', 'XEL', 'XOM', 'XPO', 'XRAY', 'XRX', 'XSPY', 'XYL', 'Y',
#                'YELP', 'YHOO', 'ZBH']

#tickers_all = ['A', 'AAN', 'AAP', 'ABTL', 'ZBH']

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


seed = 7
np.random.seed(seed)

result_str_lst = list()

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
    df_ticker.columns = ['Date', 'Open', 'Close', 'Low', 'High', 'Volume']

    ticker_data_valid = True
    for date_str in all_boersen_days:
        if not date_str in df_ticker['Date'].values:
            ticker_data_valid = False
            continue

    if not ticker_data_valid:
        for date_str in boersen_days_2018:
            result_str_lst.append([date_str + ":" + ticker_name, 0])
        print(ticker_name + " not complete")
        continue
    # df_ticker.head()

    # 2 Prepare data frame
    # 2a Price
    df_ticker['Mid_prices'] = (df_ticker.Low + df_ticker.High) / 2
    df_for_midprices = df_ticker.copy(deep=True)
    df_midprices = df_for_midprices.drop(
        ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    # df_midprices.head()

    midprices = df_midprices.values
    reframed_midprices = series_to_supervised(midprices, 0, 91)
    # reframed_midprices.head()

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
    # df_for_midprices.head()

    # 2b Volume
    df_for_volumes = df_ticker.copy(deep=True)
    df_volumes = df_for_volumes.drop(
        ['Date', 'Open', 'High', 'Low', 'Close', 'Mid_prices'], axis=1)
    # df_volumes.head()

    volumes = df_volumes.values
    reframed_volumes = series_to_supervised(volumes, 0, 91)
    # reframed_volumes.head()

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
    # df_for_volumes.head()

    # 3 Merge data frames
    merged_df = pd.merge(df_for_volumes, df_for_midprices, on='Date')
    merged_df.dropna(axis = 0, inplace = True)
    merged_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, inplace=True)
    merged_df = merged_df[~(merged_df['Entwicklungsrate Volume t+10']==np.inf)]

    # merged_df.tail()

    # 4 Add dependend variable
    df_train_label = pd.read_csv(
        join(dir_path, 'labels_train.csv'), header=0, index_col=0)
    # df_train_label.head()

    df_train_label = df_train_label.loc[:, df_train_label.columns.intersection([
        ticker_name])]
    df_train_label.columns = ['Y']
    # df_train_label.tail()

    df_complete = pd.merge(merged_df, df_train_label[['Y']], on='Date')
    df_complete = df_complete.sort_values('Date')
    #df_complete = df_complete.drop('Date', axis=1)
    # df_complete.tail()

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

    # x.shape, y.shape

    # x.max()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.7, random_state=42)
    print(y_train.shape)

    # 6 Random Forest
    rf_class = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_class.fit(x_train, y_train)

    rf_score = rf_class.score(x_test, y_test)
    print("Random Forest Score: " + ticker_name + ": " + str(rf_score))

    # type(x_test)

    # 7 Prediction for Kaggle
    # prediction = rf_class.predict(x_test)
    # prediction

    # merged_df.loc[merged_df['Date'] == '2018-01-02']

    # x = merged_df.loc[merged_df['Date'] == '2018-06-11']
    # x = x.drop(['Date'], axis=1)
    # prediction = rf_class.predict(x)
    # prediction

    # res = predictForDate('2018-06-11', merged_df, rf_class)
    # print(res)

    # date_list = daterange('2018-01-02', '2018-06-30')

    selected_dates_df = DataFrame()
    selected_dates = list()
    default_dates = list()

    for date in boersen_days_2018:
        x = merged_df.loc[merged_df['Date'] == date]
        if not x.empty:
            selected_dates_df = selected_dates_df.append(x)
            selected_dates.append(date)
        else:
            default_dates.append(date)


    selected_dates_df = selected_dates_df.drop(['Date'], axis=1)
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

    

# Transfer list to DataFrame and save
kaggle = pd.DataFrame(data=result_str_lst, columns=['Id', 'Category'])
kaggle.shape

kaggle = kaggle.to_csv('kaggle_Rapp_Katja.csv', index=False)
