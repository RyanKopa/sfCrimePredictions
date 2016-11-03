import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

np.random.seed(0)
#test and train data
dfTrain = pd.read_csv('train.csv')
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
#change categorical variable to numerical
dfAll['weekday'] = dfAll['DayOfWeek'].astype(
    'category').cat.codes.astype(float)
