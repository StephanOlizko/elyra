import pandas as pd
train_data = pd.read_csv('data/train_feateng.csv')
test_data = pd.read_csv('data/test_feateng.csv')

from sklearn.model_selection import train_test_split

X = train_data.drop('Transported', axis=1)
y = train_data['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))