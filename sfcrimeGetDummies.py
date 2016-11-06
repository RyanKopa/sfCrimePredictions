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
#change categorical variable to numerical, not using get_dummies
ohe_feats = ['DayOfWeek', 'PdDistrict']
for f in ohe_feats:
    dfAll_dummy = pd.get_dummies(dfAll[f], prefix=f)
    dfAll = dfAll.drop([f], axis=1)
    dfAll = pd.concat((dfAll, dfAll_dummy), axis=1)
dfAll['Address'] = dfAll['Address'].astype(
    'category').cat.codes.astype(float)

#splitting training and test sets
vals = dfAll.values
X = vals[:piv_train]
le = LabelEncoder()
y = le.fit_transform(labels)
X_test = vals[piv_train:]



xgb = XGBClassifier(max_depth=6, learning_rate=0.2, n_estimators=25,
                    objective='multi:softprob',
                    subsample=0.5, colsample_bytree=0.5, seed=0)

xgb.fit(X, y)
print (xgb.score(X,y))
# Can't encode with get_dummies for all categorical variables
# because too many different unique addresses values
# When dropping addresses and encoding DayOfWeek and PdDistrict
# with get_dummies, get a score of:
# score = 0.29543681503
#
# When encoded addresses with changing the categorical variable
# to float, get a score of:
# score = 0.30321200753
