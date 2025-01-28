from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('final_dataset.csv')

failed = df['failed']
df.drop(['failed'],inplace=True,axis=1)

X_train,X_test,y_train,y_test = train_test_split(df,failed,train_size=0.2,random_state=32)

def get_traning_data():
    return (X_train,y_train)

def get_validation_data():
    return (X_test,y_test)