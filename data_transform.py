import pandas as pd
train_data = pd.read_csv('data/train_preproc.csv')
test_data = pd.read_csv('data/test_preproc.csv')

numeric_features = train_data.select_dtypes(include=['float64', 'int64']).columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_data[numeric_features] = scaler.fit_transform(train_data[numeric_features])
test_data[numeric_features] = scaler.transform(test_data[numeric_features])

train_data.to_csv('data/train_transformed.csv', index=False)
test_data.to_csv('data/test_transformed.csv', index=False)