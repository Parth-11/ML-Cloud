import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('regression_data.csv')

exec_time = df['exec_time']
df.drop(['exec_time'],inplace=True,axis=1)

X_train,X_test,y_train,y_test = train_test_split(df,exec_time,test_size=0.2,random_state=42)

X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit(X_test)

def train_data() -> tuple[np.ndarray,np.ndarray]:
    return (X_train,y_train)

def test_data() -> tuple[np.ndarray,np.ndarray]:
    return (X_test,y_test)