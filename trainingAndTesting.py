###############################################################################
# I, Akinola Wilson, have read and understood the School's Academic Integrity #
# Policy, as well as guidance relating to this module, and confirm that this  #
# submission complies with the policy. The content of this file is my own     #
# original work, with any significant material copied or adapted from other   #
# sources clearly indicated and attributed.                                   #
###############################################################################
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler 
import sklearn.metrics as metrics
from sklearn.model_selection import TimeSeriesSplit

plt.style.use('ggplot')
timeSeriesSplitStrategy = TimeSeriesSplit(n_splits=12)

Train = pd.read_csv('Train.csv', index_col='Time')

Test = pd.read_csv('Test.csv', index_col='Time')

predictors = list(Train)
predictors.remove('Close')
Xtrain = Train[predictors]
Ytrain = Train[['Close']]
Xtest = Test[predictors]
Ytest = Test[['Close']]

scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Ytrain = scaler.fit_transform(Ytrain)
Xtest = scaler.fit_transform(Xtest)
Ytest = scaler.fit_transform(Ytest)
###############################################################################
# 
# Hyperparameter Evaluation and Selection
#
###############################################################################
alphaLASSO = np.arange(0.001, 0.02, 0.0001)
alphaMLP = np.arange(0.55,0.75+0.001,0.001)
kRange = np.arange(100, 200, 1)
alphaRidge = np.arange(150, 250+0.01, 1) 
scoreAlphaLASSO = []
scoreAlphaMLP = []
scoreAlphaKNN = []
scoreAlphaRidge = []

for aL in alphaLASSO:
    LASSO = linear_model.Lasso(alpha=aL, max_iter=10000)
    scoresLASSO = cross_val_score(LASSO, Xtrain, Ytrain, cv=timeSeriesSplitStrategy,
                                  scoring='neg_mean_squared_error')
    scoreAlphaLASSO.append(scoresLASSO.mean())

for aM in alphaMLP:
    MLPReg = MLPRegressor(alpha=aM, max_iter=500)
    scoresMLP = cross_val_score(MLPReg, Xtrain, np.ravel(Ytrain),
                                cv=timeSeriesSplitStrategy,
                                scoring='neg_mean_squared_error')
    scoreAlphaMLP.append(scoresMLP.mean())

for aK in kRange:
    KNNreg = KNeighborsRegressor(n_neighbors=aK, weights='distance')
    scoresKNN = cross_val_score(KNNreg, Xtrain, Ytrain,
                                cv=timeSeriesSplitStrategy,
                                scoring='neg_mean_squared_error')
    scoreAlphaKNN.append(scoresKNN.mean())
for aR in alphaRidge:
    Ridge = linear_model.Ridge(alpha=aR)
    scoresRidge = cross_val_score(Ridge, Xtrain, Ytrain, cv=timeSeriesSplitStrategy,
                                  scoring='neg_mean_squared_error') 
    scoreAlphaRidge.append(scoresRidge.mean())

 
com1 = pd.DataFrame({'Alpha':alphaLASSO,'MSE':scoreAlphaLASSO})
com1.set_index('Alpha', inplace=True)
alphaBestLASSO = com1[com1['MSE']==max(com1['MSE'])].index.values.astype(float)[0]    
com2 = pd.DataFrame({'Alpha':alphaMLP,'MSE':scoreAlphaMLP})
com2.set_index('Alpha', inplace=True)
alphaBestMLP = com2[com2['MSE']==max(com2['MSE'])].index.values.astype(float)[0]
com3 = pd.DataFrame({'Number of Neighbours':kRange,'MSE':scoreAlphaKNN})
com3.set_index('Number of Neighbours', inplace=True)
kBest = com3[com3['MSE']==max(com3['MSE'])].index.values.astype(int)[0]
com4 = pd.DataFrame({'Alpha':alphaRidge,'MSE':scoreAlphaRidge})
com4.set_index('Alpha', inplace=True)
alphaBestRidge = com4[com4['MSE']==max(com4['MSE'])].index.values.astype(float)[0]
###############################################################################
# 
# Hyperparameter Selection Plotting
#
###############################################################################
fig = plt.figure(figsize=(20,12))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.plot(alphaLASSO, (-1)*np.array(scoreAlphaLASSO))
ax2.plot(alphaMLP, (-1)*np.array(scoreAlphaMLP))
ax3.plot(kRange,(-1)*np.array(scoreAlphaKNN))
ax4.plot(alphaRidge, (-1)*np.array(scoreAlphaRidge)) 
ax1.set_xlabel(r'$\alpha$')
ax1.set_ylabel('Mean Square Error: LASSO')
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel('Mean Square Error: MLP')
ax3.set_xlabel(r'$\alpha$')
ax3.set_ylabel('Mean Square Error: KNN')
ax4.set_xlabel(r'$k$')
ax4.set_ylabel('Mean Square Error: Ridge')
ax1.axvline(x=alphaBestLASSO, ls='--', color='blue')
ax1.set_title(r' Optimised $\alpha$, $\alpha$* = {}'.format(alphaBestLASSO)) 
ax2.axvline(x=alphaBestMLP, ls='--', color='blue')
ax2.set_title(r' Optimised $\alpha$, $\alpha$* = {}'.format(alphaBestMLP))  
ax3.axvline(x=kBest, ls='--', color='blue')
ax3.set_title(r' Optimised number of nearest neighbours $k$, $k$* = {}'.format(int(kBest))) 
ax4.axvline(x=alphaBestRidge, ls='--', color='blue')
ax4.set_title(r' Optimised $\alpha$, $\alpha$* = {}'.format(alphaBestRidge))
###############################################################################
# 
# Model Fitting and Testing 
#
###############################################################################
Ytest = scaler.inverse_transform(Ytest)

LASSO = linear_model.Lasso(alpha=alphaBestLASSO, max_iter= 10000) # Hyperparamter 0.0104
LASSO.fit(Xtrain,Ytrain)

MLP = MLPRegressor(alpha=alphaBestMLP, max_iter=500) # Hyperparamter alpha=0.442
MLP.fit(Xtrain,Ytrain)

KNN = KNeighborsRegressor(n_neighbors=kBest, weights='distance') # kBest = 261
KNN.fit(Xtrain,Ytrain)

Ridge = linear_model.Ridge(alpha=alphaBestRidge) # Hyperparamter best = 179
Ridge.fit(Xtrain,Ytrain)

LASSOtest = LASSO.predict(Xtest)
LASSOtest = np.array(LASSOtest) 
LASSOtest = np.expand_dims(LASSOtest, axis=1)
LASSOtest = scaler.inverse_transform(LASSOtest)

MLPtest = MLP.predict(Xtest)
MLPtest = np.expand_dims(MLPtest, axis=1)
MLPtest = scaler.inverse_transform(MLPtest)

KNNtest = np.ravel(KNN.predict(Xtest))
KNNtest = np.expand_dims(KNNtest, axis=1)
KNNtest = scaler.inverse_transform(KNNregtest)

Ridgetest = Ridge.predict(Xtest)
Ridgetest = scaler.inverse_transform(Ridgetest)
###############################################################################
# 
# Plotting Testing Performance vs. Actual Closing Price 
#
###############################################################################
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,1)
ax3 = fig.add_subplot(2,2,1)
ax4 = fig.add_subplot(2,2,1)

ax1.plot(np.arange(0, np.shape(Xtest)[0]), LASSOtest, 'b-')
ax1.plot(np.arange(0, np.shape(Xtest)[0]), Ytest, 'r-' ) 
ax1.legend(('Predicted','Actual'))
ax2.plot(np.arange(0, np.shape(Xtest)[0]), MLPtest, 'b-') 
ax2.plot(np.arange(0, np.shape(Xtest)[0]), Ytest, 'r-' )
ax2.legend(('Predicted','Actual'))
ax3.plot(np.arange(0, np.shape(Xtest)[0]), KNNtest, 'b-')
ax3.plot(np.arange(0, np.shape(Xtest)[0]), Ytest, 'r-' )
ax3.legend(('Predicted','Actual'))
ax4.plot(np.arange(0, np.shape(Xtest)[0]), Ridgetest, 'b-')
ax4.plot(np.arange(0, np.shape(Xtest)[0]), Ytest, 'r-' )
ax4.legend(('Predicted','Actual'))

print('On test, MSE for LASSO: {}'.format( metrics.mean_squared_error(Ytest,LASSOtest)))
print('On test, MSE for MLP: {}'.format( metrics.mean_squared_error(Ytest,MLPtest))
print('On test, MSE for KNN: {}'.format( metrics.mean_squared_error(Ytest,KNNtest)))
print('On test, MSE for Ridge: {}'.format( metrics.mean_squared_error(Ytest,Ridgetest))) 
###############################################################################
# 
# Gathering Test and Training Data in Data Frame 
#
###############################################################################
models = ['LASSO', 'MLP', 'KNN', 'Ridge']
trainingErrorAbs = [metrics.mean_absolute_error(Ytrain,LASSO.predict(Xtrain)),
                    metrics.mean.absolute.error(Ytrain,MLP.predict(Xtrain)), 
                    metrics.mean.absolute.error(Ytrain,KNN.predict(Xtrain)),
                    metrics.mean.absolute.error(Ytrain,Ridge.predict(Xtrain)) ]

trainingErrorRMSE = [(metrics.mean_squared_error(Ytrain,LASSO.predict(Xtrain)))**(1/2),
                     (metrics.mean_squared_error(Ytrain,MLP.predict(Xtrain)))**(1/2),
                     (metrics.mean_squared_error(Ytrain,KNN.predict(Xtrain)))**(1/2),
                     (metrics.mean_squared_error(Ytrain,Ridge.predict(Xtrain)))**(1/2)]

dfTrainResults = pd.DataFrame({'Model':models,
                               'Training Absolute Mean Error':trainingErrorAbs,
                               'Training Root Mean Square Error':trainingErrorRMSE })
    
testErrorAbs = [metrics.mean_absolute_error(Ytest,LASSO.predict(Xtest)),
                metrics.mean.absolute.error(Ytest,MLP.predict(Xtest)),
                metrics.mean.absolute.error(Ytest,KNN.predict(Xtest)),
                metrics.mean.absolute.error(Ytest,Ridge.predict(Xtest)) ]

testErrorRMSE = [(metrics.mean_squared_error(Ytest,LASSO.predict(Xtest)))**(1/2),
                 (metrics.mean_squared_error(Ytest,MLP.predict(Xtest)))**(1/2),
                 (metrics.mean_squared_error(Ytest,KNN.predict(Xtest)))**(1/2),
                 (metrics.mean_squared_error(Ytest,Ridge.predict(Xtest)))**(1/2)]
dfTestResults pd.DataFrame({'Model':models,
                            'Test Absolute Mean Error':testErrorAbs,
                            'Test Root Mean Square Error': testErrorRMSE })

