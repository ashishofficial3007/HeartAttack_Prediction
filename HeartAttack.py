import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import statistics
from sklearn.linear_model import LogisticRegression


import time

import warnings
warnings.filterwarnings(action="ignore")

pd.set_option('display.width', 1000)
pd.set_option('display.max_column', 15)

data = pd.read_csv('HeartAttack_data.csv', index_col=False, na_values=["?"])
print("\n\n\nSample HeartAttack dataset head(5) :- \n\n", data.head(5) )

print("\n\n\nShape of the HeartAttack dataset  data.shape = ", end="")
print( data.shape)
#(294,14)

print("\n\n\nHeartAttack data decription : \n")
print( data.describe() )
#No missing data


plt.hist(data['num'])
plt.title('num (Yes=1 , No=0)')
plt.show()



data = data.fillna(data.median())

data.drop(data.index[91])
data.drop(data.index[279])
del data['ca']
print("\n\nAfter Deletion of 'Unnamed: 14' column\n", data)



print("\n\n\ndata.groupby('num').size()\n")
print(data.groupby('num').size())

#num
#0    188
#1    106
#dtype: int64


data.plot(kind='density', subplots=True, layout=(5,5), sharex=False, legend=False, fontsize=1)
plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.imshow(data.corr() )
ax1.grid(True)
plt.title('Heart Attack Attributes Correlation')
fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
plt.show()




Y = data['num'].values
X = data.drop('num', axis=1).values

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.33, random_state=21)




models_list = []
models_list.append(('LOG', LogisticRegression()))
models_list.append(('LDA', LinearDiscriminantAnalysis()))
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('SVM', SVC()))
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))
#without standerdisation

num_folds = 10

results = []
names = []

for name, model in models_list:
    kfold = KFold(n_splits=num_folds, random_state=221)
    startTime = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    endTime = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), endTime-startTime))


fig = plt.figure()
fig.suptitle('Performance Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()




pipelines = []
pipelines.append(('ScaledLOG', Pipeline([('Scaler', StandardScaler()),('LOG', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
#standerdise and then apply the model



results = []
names = []



print("\n\n\nAccuracies of algorithm after scaled dataset\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds, random_state=221)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='roc_auc')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))


fig = plt.figure()
fig.suptitle('Performance Comparison after Scaled Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



model = LogisticRegression()
start = time.time()
model.fit(X_train, Y_train)
end = time.time()
print( "\n\nLogistic Regression Training Completed. It's Run Time: %f" % (end-start))



predictions = model.predict(X_test)
print("All predictions done successfully by LogarithmicRegression Machine Learning Algorithms")
print("\n\nAccuracy score %f" % accuracy_score(Y_test, predictions))

print("\n\n")
print("confusion_matrix = \n")
print( confusion_matrix(Y_test, predictions))




from sklearn.externals import joblib
filename =  "finalized_HeartAttack_model.sav"
joblib.dump(model, filename)
print( "Best Performing Model dumped successfully into a file by Joblib")
#save the trained data

print()

print("ML Solution Proposed by: Ashish Singh")
print("Email id: ashishofficial3007@gmail.com")
print("Roll number: 1806291")