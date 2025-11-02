
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
filename="energy_meter.csv"
names=['voltage','current','power','class']
dataset=read_csv(filename,names=names, skiprows=1)
dataset[['voltage', 'current', 'power']] = dataset[['voltage', 'current', 'power']].apply(pd.to_numeric, errors='coerce')
dataset.dropna(inplace=True)
array=dataset.values
x=array[:,0:3]
y=array[:,3]
x_train,x_validation,y_train,y_validation=train_test_split(x,y,test_size=0.2,random_state=1)
model=SVC(gamma='auto')
model.fit(x_train,y_train)
result=model.score(x_validation,y_validation)
print(result)
value=[[212.7316,0.84753,180.296343]]
prediction=model.predict(value)
print(prediction[0])
