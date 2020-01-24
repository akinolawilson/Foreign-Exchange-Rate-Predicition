###############################################################################
# I, Akinola Wilson, have read and understood the School's Academic Integrity #
# Policy, as well as guidance relating to this module, and confirm that this  #
# submission complies with the policy. The content of this file is my own     #
# original work, with any significant material copied or adapted from other   #
# sources clearly indicated and attributed.                                   #
###############################################################################
from datetime import datetime
import pandas as pd 
import numpy as np
from pytz import all_timezones
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression as mir
from sklearn.feature_selection import VarianceThreshold 

###############################################################################
#
#
#                     Preparing Currency data files
#
#
###############################################################################
midnight = '\n00:00'
currencies = ['GBP', 'CAD', 'EUR', 'NZD', 'CHF', 'SEK']
startcol = range(0, 6 * len(currencies), 6)
ncol = 5
dfCurs = dict()
for cur, i in zip(currencies, startcol):
    dfCur = pd.read_csv('USD_EUR_GBP_CAD_CHF_SEK_SNZ.csv',
                     header=3, usecols=range(i, i+ncol),
                     names=('Time', 'Open', 'Close', 'High', 'Low'))
    
    dfCur = dfCur.dropna()
    for i in range(np.shape(dfCur)[0]):
        if len(str(dfCur.loc[i,'Time'])) <= 10:
            dfCur.loc[i,'Time'] = str(dfCur.loc[i,'Time']) + str(midnight)
            
    dfCur['Time'] = pd.to_datetime(dfCur['Time'], format='%d/%m/%Y\n%H:%M')
    dfCur = dfCur.set_index('Time')
    dfCur = dfCur.tz_localize('Etc/GMT-0')
    dfCur = dfCur.tz_convert('America/New_York')
    dfCurs[cur] = dfCur
    
# Saving data
#for i in currencies:
    
#    dfs[i].to_csv('{}History.csv'.format(i))
    
################################################################################
#
#    
#                     Currency swaps data preparation 
#
# Design choice: Repeatinf Currency Swap rates with equal weight across all tick
# intervals to align with USDvsGBP closing price tick intervals   
################################################################################
dfSwap = pd.read_csv('CurrencySwaps.csv')
dfSwap = dfSwap.drop(index=[0,1,2,4,5])
dfSwap = dfSwap.rename(columns={'Start Date':'Time', '03/06/2019':'GBP Swaps',
                                'Unnamed: 2':'USD Swaps'})
dfSwap.reset_index(drop=True, inplace=True)
dfSwap.at[0,'GBP Swaps'] = 0.84604
dfSwap.at[0, 'USD Swaps'] = 2.1563
dfSwap.at[0, 'Time'] = '03/06/2019\n00:00'
dfSwap['Time'] = pd.to_datetime(dfSwap['Time'], format='%d/%m/%Y\n%H:%M')

patternForRepeatingVals = np.array([240,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,108,
                                  288,288,288,288,204,120, #<-notice day light saving
                                  288,288,288,288,204])    #  causes extra hour 
                                                           #  (60/5mins=12)-> 108+12 =120
# repeating values for sundays
missingGBPswapRate = []
missingUSDswapRate = []
missingDates = []
index = np.arange(4.5, 94.5, 5)

for j in np.arange(5, 95, 5):
    missingGBPswapRate.append(dfSwap['GBP Swaps'][j])
    missingUSDswapRate.append(dfSwap['USD Swaps'][j])  
# Obtaining strings of dates that need to be entered into dataframe 
i = 0 
while i < 1:
    for june in range(4):
        if june == 0:
            missingDates.append('2019-06-09\n 00:00:00')
        if june != 0:
            missingDates.append('2019-06-' + str( int(9 + 7*june) ) + '\n 00:00:00')
    i+=1
while i == 1:
    for july in range(4):
        if july == 0:
            missingDates.append('2019-07-07\n 00:00:00')
        if july != 0:
            missingDates.append('2019-07-' + str( int(7 + 7*july) ) + '\n 00:00:00')
    i+=1
while i == 2:
    for aug in range(4):
        if aug == 0:
            missingDates.append('2019-08-04\n 00:00:00')
        if aug != 0:
            missingDates.append('2019-08-' + str( int(4 + 7*aug) ) + '\n 00:00:00')
    i+=1       
while i == 3:
    for sep in range(4):
        if sep == 0:
            missingDates.append('2019-09-01\n 00:00:00')
        if sep == 1:
            missingDates.append('2019-09-08\n 00:00:00')
        if sep != 0 and sep != 1:
            missingDates.append('2019-09-' + str( int(1 + 7*sep) ) + '\n 00:00:00')
        if sep != 0 and sep != 1 and sep != 2:
            missingDates.append('2019-09-' + str( int(8 + 7*sep) ) + '\n 00:00:00')
    i+=1
if i == 4:
    missingDates.append('2019-10-06\n 00:00:00')


    
for i in range(len(index)):
    a = pd.DataFrame({'Time':missingDates[i],'GBP Swaps':missingGBPswapRate[i],
                      'USD Swaps':missingUSDswapRate[i]}, index=[index[i]])
    dfSwap = dfSwap.append(a, ignore_index=False)
    
dfSwap = dfSwap.sort_index()
dfSwap = dfSwap.reset_index(drop=True)
dfSwap =  dfSwap.loc[dfSwap.index.repeat(patternForRepeatingVals)].reset_index(drop=True)
toBedropped = np.arange(27624, np.shape(dfSwap)[0]) 
dfSwap = dfSwap.drop(index=toBedropped)
dfSwap.index = dfCurs['GBP'].index
dfSwap = dfSwap.drop(columns='Time')
###############################################################################

#                          Libor data preparation 

###############################################################################
liborColnames =['Time', 'LIBOR over Night UK','LIBOR over Night US']
dflib = pd.read_csv('liborOverNight.csv',header=5,usecols=[0,1,2], names=liborColnames )


dflib.at[0,'LIBOR over Night US'] = 2.36
dflib.at[0,'Time'] = '03/06/2019'

LiborUK = []
LiborUS = []
for j in np.arange(5, 95, 5):
    LiborUK.append(dflib['LIBOR over Night UK'][j])
    LiborUS.append(dflib['LIBOR over Night US'][j])
for i in range(len(index)):
    a = pd.DataFrame({'Time':missingDates[i],'LIBOR over Night UK':LiborUK[i],
                      'LIBOR over Night US':LiborUS[i]}, index=[index[i]])
    dflib = dflib.append(a, ignore_index=False)
    

dflib = dflib.sort_index()
dflib = dflib.reset_index(drop=True)

dflib =  dflib.loc[dflib.index.repeat(patternForRepeatingVals)].reset_index(drop=True)
toBedropped = np.arange(27624, np.shape(dflib)[0])
dflib = dflib.drop(index=toBedropped)
dflib.index = dfCurs['GBP'].index
dflib.drop(columns='Time',inplace=True)
###############################################################################

#                        Indicator preparations

###############################################################################

midnight = '\n00:00'
startcol = [0, 3, 7, 10, 13, 16, 20, 23]
ncol = [2,3,2,2,2,3,2,2]
index = [2772.5, 18876.5]
date = ['16/06/2019\n20:05','01/09/2019\n20:05']
                    #    RSI        Mom  & Mom Avg   exp Mov Avg   willams % R
additionalValues = [(59.50321862, (0.0004,-0.00026), 0.794665459, -22.22222222),
                    (50.62702489, (0.0004, 0.00022), 0.822064806, -22.22222222)]

additional = list(zip(index, date, additionalValues))

indicators = ['RSIb', 'momMovAvgb','expAvgb', 'W%Rb', 
              'RSIa', 'momMovAvga', 'expAvga','W%Ra']

names = [('Time', 'RSI Bid'), ('Time','Momentum Bid','Moving Avg Bid'),
         ('Time', 'Exponential Moving Avg Bid'), ('Time','William\'s % Range Bid'),
         ('Time', 'RSI Ask'), ('Time','Momentum Ask','Moving Avg Ask'),
         ('Time', 'Exponential Moving Avg Ask'), ('Time','William\'s % Range Ask')]

dfInds = dict()

for tech, colindex, colnames, numOfcols in zip(indicators, startcol, names, ncol):
    dfInd = pd.read_csv('IndicatorsMomentumRSIExpMovAvgWilliams.csv',
                     header=2, usecols=range(colindex, colindex + numOfcols),
                     names=colnames)
    
    for i in range(np.shape(dfInd)[0]):
        if len(str(dfInd.loc[i,'Time'])) <= 10:
            dfInd.loc[i,'Time'] = str(dfInd.loc[i,'Time']) + str(midnight)
            
    if tech == 'RSIb':
        dfInd.fillna(60.0, inplace=True)
        dfInd = dfInd.append(pd.DataFrame({'Time': additional[1][1],
                                           'RSI Bid':additional[1][2][0]},
                                            index = [additional[1][0]]),
                                            ignore_index=False)
        dfInd = dfInd.sort_index()
    if tech == 'RSIa':
        dfInd.fillna(60.0, inplace=True)
        dfInd = dfInd.append(pd.DataFrame({'Time': additional[0][1],
                                            'RSI Ask':additional[0][2][0]},
                                            index = [additional[0][0]]),
                                            ignore_index=False)
        dfInd = dfInd.sort_index()
    if tech == 'momMovAvgb':
        dfInd.fillna(0.0004, inplace=True)
        dfInd.replace(-999999, 0.0005, inplace=True)
        dfInd = dfInd.append(pd.DataFrame({'Time': additional[1][1],
                                           'Momentum Bid':additional[1][2][1][0],
                                           'Moving Avg Bid':additional[1][2][1][1]},
                                            index = [additional[1][0]]),
                                            ignore_index=False)
        dfInd = dfInd.sort_index()     
    if tech == 'momMovAvga':
        dfInd.fillna(0.0004, inplace=True)
        dfInd.replace(-999999, 0.0005, inplace=True)
        dfInd = dfInd.append(pd.DataFrame({'Time': additional[0][1],
                                           'Momentum Ask':additional[0][2][1][0],
                                           'Moving Avg Ask':additional[0][2][1][1]},
                                            index = [additional[0][0]]),
                                            ignore_index=False)
        dfInd = dfInd.sort_index()
    if tech == 'W%Rb':
        dfInd.fillna(-13.33333333, inplace=True)
        dfInd = dfInd.append(pd.DataFrame({'Time': additional[1][1],
                                           'William\'s % Range Bid':additional[1][2][3]},
                                            index = [additional[1][0]]),
                                            ignore_index=False)
        dfInd = dfInd.sort_index()
    if tech == 'W%Ra':
        dfInd.fillna(-13.33333333, inplace=True)
        dfInd = dfInd.append(pd.DataFrame({'Time': additional[0][1],
                                           'William\'s % Range Ask':additional[0][2][3]},
                                            index = [additional[0][0]]),
                                            ignore_index=False)
        dfInd = dfInd.sort_index()
    if tech == 'expAvgb':
        
        dfInd = dfInd.append(pd.DataFrame({'Time': additional[1][1],
                                           'Exponential Moving Avg Bid':additional[1][2][2]},
                                            index = [additional[0][0]]),
                                            ignore_index=False)
        dfInd = dfInd.sort_index()
    if tech == 'expAvga':
        
        dfInd = dfInd.append(pd.DataFrame({'Time': additional[0][1],
                                           'Exponential Moving Avg Ask':additional[0][2][2]},
                                            index = [additional[1][0]]),
                                            ignore_index=False)
        dfInd = dfInd.sort_index()        
    dfInd.drop(dfInd.tail(27).index, inplace=True)                
    dfInd['Time'] = pd.to_datetime(dfInd['Time'], format='%d/%m/%Y\n%H:%M')
    dfInd = dfInd.set_index('Time')
    dfInd = dfInd.tz_localize('Etc/GMT-0')
    dfInd = dfInd.tz_convert('America/New_York')
    dfInds[tech] = dfInd
###############################################################################
#   
#
#                          T - Bill data preparation 
#
# Design choice: Repeat values of Tbill data with equal weight over all tick 
# intervals
###############################################################################
indices = np.arange(4.5, 94.5, 5)
    
columnNames = ('Time','UK generic 1 month T-Bill Yield',
               'US generic 1 month T-Bill Yield')
dfTbill = pd.read_csv('TBillYield1MonthUSAndUK.csv', header=5, usecols=[0,1,2],
                      names=columnNames)
dfTbill.loc[0,'Time'] = '03/06/2019'
dfTbill.loc[0, 'US generic 1 month T-Bill Yield'] = 2.337
dfTbill['Time'] = pd.to_datetime(dfTbill['Time'], format='%d/%m/%Y')

TbillUK = []
TbillUS = []

for j in np.arange(5, 95, 5):
    TbillUK.append(  dfTbill['UK generic 1 month T-Bill Yield'][j]  )
    TbillUS.append(  dfTbill['US generic 1 month T-Bill Yield'][j]  )
    
for i in range(len(indices)):    
    a = pd.DataFrame({'Time': missingDates[i], 'UK generic 1 month T-Bill Yield': TbillUK[i],
                      'US generic 1 month T-Bill Yield': TbillUS[i]}, index=[ indices[i] ])
    
    dfTbill = dfTbill.append(a, ignore_index=False)

dfTbill = dfTbill.sort_index()
dfTbill = dfTbill.reset_index(drop=True)
dfTbill =  dfTbill.loc[dfTbill.index.repeat(patternForRepeatingVals)].reset_index(drop=True)
dfTbill.drop(dfTbill.tail(48).index, inplace=True)  
dfTbill.index = dfCurs['GBP'].index
dfTbill = dfTbill.drop(columns='Time')



dfDerivatives = dict({'swaps':dfSwap,'tbill':dfTbill,'libor':dflib})
################################################################################
#
#
#                    Google analytics data preparation 
#
# Design choice: repeat search frequencies over each 5 minute tick interval with  
# equal weight. Disregarding weekend searches 
#
###############################################################################
dfbritishPoundVsUSdollar = pd.read_csv('searchGoogleHistorybritishPoundVsUSdollar.csv') 
dfGBPvsUSD = pd.read_csv('searchGoogleHistoryGBPvsUSD.csv')
dfTravellingUK = pd.read_csv('searchGoogleHistoryTravellingUK.csv')
dfTravellingUS = pd.read_csv('SearchHistoryTravellingUS.csv')
toBeDropped_googleFreqWeekends = np.arange(5, 131, 7)
#toBedropped_tails = np.arange(27624, np.shape(dfGBPvsUSD)[0])
googleColumnNames = [('Time','Search: British Pound Vs US Dollar freq'),
                     ('Time','Search: GBP vs USD freq'),('Time', 'Search: Travelling UK freq'),
                     ('Time','Search: Travelling US freq')]
k = ('gbpvsdollar', 'GBPvsUSD','TravUK', 'TravUS')
dfGoogleFreqs = dict()
for name,df, keys in zip((googleColumnNames) ,(dfbritishPoundVsUSdollar, dfGBPvsUSD,
                         dfTravellingUK, dfTravellingUS), k):
    df.drop(index=[0,1], inplace=True)
    df.columns = name
    df.reset_index(drop=True, inplace=True)
    df.drop(index=toBeDropped_googleFreqWeekends, inplace=True)
    dfGoogleFreqs[keys] = df 

for keys in k:
    dfGoogleFreqs[keys] = dfGoogleFreqs[keys].loc[dfGoogleFreqs[keys]\
                  .index.repeat(patternForRepeatingVals)].reset_index(drop=True)
    
    toBedropped_tails = np.arange(27624, np.shape(dfGoogleFreqs[keys])[0])
    dfGoogleFreqs[keys].drop(index=toBedropped_tails, inplace=True)
    dfGoogleFreqs[keys].index = dfCurs['GBP'].index
    dfGoogleFreqs[keys].drop(columns='Time', inplace=True)

###############################################################################
#   
#
# Design choice: Consider mean of bid and ask side data of  all indicators,
# derivatives, and other features which have dual-side aspect.    
#
#
###############################################################################
dfIndsMean = dict()
indictorsMean = ['RSI','momMovAvg','expAvg', 'W%R']
indicatorsPairedKeys = [('RSIb','RSIa'), ('momMovAvgb','momMovAvga'),
                        ('expAvgb', 'expAvga'), ('W%Rb','W%Ra')]
for newKeys, oldKeys in zip(indictorsMean,indicatorsPairedKeys):
    a = pd.concat((dfInds[oldKeys[0]],dfInds[oldKeys[1]]), axis=1)
    b = pd.DataFrame({newKeys: a.mean(axis=1)})
    dfIndsMean[newKeys] = b
    if newKeys == 'momMovAvg':
            amom = pd.concat((dfInds[oldKeys[0]]['Momentum Bid'],
                              dfInds[oldKeys[1]]['Momentum Ask']), axis=1)
            
            amov = pd.concat((dfInds[oldKeys[0]]['Moving Avg Bid'],
                              dfInds[oldKeys[1]]['Moving Avg Ask']), axis=1)
            
            b = pd.DataFrame({'Momentum': amom.mean(axis=1),
                              'Moving Avg':amov.mean(axis=1)})
            dfIndsMean[newKeys] = b
        
###############################################################################
            
# Data compiling; Creating overall data matrix, splitting into test and training

            
###############################################################################
            
TimeSeries = pd.concat([dfCurs['GBP'],dfIndsMean['momMovAvg'],dfIndsMean['expAvg'],
                        dfIndsMean['RSI'],dfIndsMean['W%R'], dfDerivatives['swaps'],
                        dfDerivatives['tbill'],dfDerivatives['libor'],
                        dfGoogleFreqs['GBPvsUSD'],dfGoogleFreqs['TravUK'],
                        dfGoogleFreqs['TravUS']] ,axis=1)

Train = TimeSeries['2019-06-03':'2019-08-30']
Test = TimeSeries['2019-09-01':]


###############################################################################
#                                                                             #
#                                                                             #       
#                   Data exploration and feature selection                    #
#                                                                             #
#                                                                             #
###############################################################################


###############################################################################
#
# Heatmap of features 
#
###############################################################################
plt.style.use('ggplot')
correlationMap = Train.corr()
topCorrelatedFeatures = correlationMap.index
plt.figure(figsize=(20,15))
plt.axes([0.1, 0.30, 0.7, 0.65])
sns.heatmap(correlationMap, annot=True, cmap='bwr_r', square=True, fmt='.2f')

###############################################################################
#
# Pearson's Correlation Coefficient  
#
###############################################################################

features = list(Train.drop(columns='Close'))
Xpearson = Train.drop(columns='Close')
Ypearson = Train.Close
Xpearson = Xpearson.to_numpy().astype(float)
Ypearson = Ypearson.to_numpy()
corrcoefficient = []

for i in range(len(features)):
    corrcoefficient.append( np.corrcoef( Xpearson[:,i], Ypearson)) 
      
d = []
val = [] 

for i in range(len(features)):
    a = str(features[i])
    b = format(corrcoefficient[i][0,1], '.3f')
    val.append(b)
    individualCoef = [a,b]
    d.append(individualCoef)

# plotting   
fig = plt.figure(figsize=(10,12))
plt.axes([0.15, 0.30, 0.7, 0.55])

axPearson = fig.gca()

val = [float(x) for x in val]
plt.bar(np.arange(1,18,1),val,tick_label=features)
plt.xticks(rotation=90)
axPearson.set_ylim((-1.05,1.05))
axPearson.set_title('Pearson\'s Correlation Coefficient with Closing Price',
                    fontsize=14)
###############################################################################
#
# Mutual Info Regression  
#
###############################################################################

Xinfo = Train.drop(columns='Close')
Yinfo = Train.Close 
mutinfo = mir(Xinfo,Yinfo.values.flatten(), n_neighbors = 10)
fig = plt.figure(figsize=(10,12))
plt.axes([0.15, 0.30, 0.7, 0.55])
ax = fig.gca()
plt.bar(np.arange(1,18,1),mutinfo,tick_label=features)
plt.xticks(rotation=90)
ax.set_ylabel('Information, nat')
ax.set_title('Mutual Information of Features with respect to Closing Price',
             fontsize=14)


# Based on analysis, will exclude features:

features.remove('Open')
features.remove('Low')
features.remove('expAvg')
features.remove('Search: Travelling UK freq')
features.remove('Search: Travelling US freq') 
features = ['Close'] + features
Train = Train[features]
Test = Test[features]



