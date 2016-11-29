# www.kaggle.com/keldibek/sf-crime/xgboost-crime-classification/notebook
#
#Python Model used to predict the classification of a crime, given the location and time of a crime scene
#based on the data provided by Kaggle.
#
#Input: train.csv and test.csv provided by Kaggle
#Output: submission.csv, a csv file that lists the id of each reported crime and a list of probabilities 
#of the likeliness that the reported crime fits a given classification.
#
#Notes for future improvements:
#Cross validation for building the model
#Iteration mode to determine the best depth for the model (best number for num_round)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.cross_validation import train_test_split

np.random.seed(0)
#test and train data
dfTrain = pd.read_csv('train.csv')
dfTrain = dfTrain.sort_values(by=u'Category', ascending=1)

dfTest = pd.read_csv('test.csv')

#categories predicted for
labels = dfTrain[u'Category'].values
#drop data not used in model
dfTrain = dfTrain.drop([u'Category'], axis=1)
dfTrain = dfTrain.drop([u'Descript'], axis = 1)
dfTrain = dfTrain.drop([u'Resolution'], axis = 1)
#test id's for model
idTest = dfTest[u'Id']
dfTest = dfTest.drop([u'Id'], axis = 1)
piv_train = dfTrain.shape[0]

#combine testing and training sets for data wrangling
dfAll = pd.concat((dfTrain, dfTest), axis=0, ignore_index=True)

dateTime = np.vstack(dfAll.Dates.astype(str).apply(
    lambda x: list(map(float, x.replace('-',' ')
        .replace(':', ' ').split(' ')))).values)
#change time and date to a more continuous format
dfAll['year'] = dateTime[:,0]
dfAll['month'] = dateTime[:,1]
dfAll['day'] = dateTime[:,2]
dfAll['hour'] = dateTime[:,3]
dfAll['minute'] = dateTime[:,4]
dfAll['second'] = dateTime[:,5]
#remove date and time account created
dfAll = dfAll.drop(['Dates'], axis=1)
#change categorical variable to numerical, not using get_dummies
dfAll['Weekday'] = dfAll['DayOfWeek'].astype(
    'category').cat.codes.astype(float)
dfAll = dfAll.drop(['DayOfWeek'], axis = 1)
dfAll['PdDistrict'] = dfAll['PdDistrict'].astype(
    'category').cat.codes.astype(float)
dfAll['Address'] = dfAll['Address'].astype(
    'category').cat.codes.astype(float)


def set_param():
    # setup parameters for xgboost
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.4
    param['silent'] = 0
    param['nthread'] = 4
    param['num_class'] = len(np.unique(labels))
    param['eval_metric'] = 'mlogloss'
    # Model complexity
    param['max_depth'] = 8 #set to 8
    param['min_child_weight'] = 1
    param['gamma'] = 0
    param['reg_alfa'] = 0.05
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8 #set to 1
    # Imbalanced data
    param['max_delta_step'] = 1
    return param

#splitting training and test sets
vals = dfAll.values
X = vals[:piv_train]
X_test = vals[piv_train:]
le = LabelEncoder()
y = le.fit_transform(labels)

trainMatrix = xgb.DMatrix(X, label = y)
testMatrix = xgb.DMatrix(X_test)

num_class = len(np.unique(labels))

param = set_param()
watchlist = [(trainMatrix,'trainEval')]
num_round = 10

#train xgb
model = xgb.train(param, trainMatrix, num_round, watchlist);
yprob = model.predict(testMatrix).reshape(X_test.shape[0], num_class)
# ylabel = np.argmax(yprob, axis = 1)

ids = np.linspace(0,X_test.shape[0],X_test.shape[0], dtype = int)

#Generate submission
finalLabels = np.insert(np.unique(labels), 0, 'Id')
sub = pd.DataFrame(np.column_stack((ids, yprob)),
    columns=finalLabels)
sub.Id = sub.Id.astype(int)
sub.to_csv('submission.csv',index=False)
