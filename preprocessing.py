import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

train_data['HomePlanet'].fillna(train_data['HomePlanet'].mode()[0], inplace=True)
test_data['HomePlanet'].fillna(train_data['HomePlanet'].mode()[0], inplace=True)

train_data['CryoSleep'].fillna(train_data['CryoSleep'].mode()[0], inplace=True)
test_data['CryoSleep'].fillna(train_data['CryoSleep'].mode()[0], inplace=True)

train_data['Destination'].fillna(train_data['Destination'].mode()[0], inplace=True)
test_data['Destination'].fillna(train_data['Destination'].mode()[0], inplace=True)

train_data['Age'].fillna(train_data['Age'].mode()[0], inplace=True)
test_data['Age'].fillna(train_data['Age'].mode()[0], inplace=True)

train_data['VIP'].fillna(train_data['VIP'].mode()[0], inplace=True)
test_data['VIP'].fillna(train_data['VIP'].mode()[0], inplace=True)


train_data['RoomService'].fillna(train_data['RoomService'].median(), inplace=True)
train_data['FoodCourt'].fillna(train_data['FoodCourt'].median(), inplace=True)
train_data['ShoppingMall'].fillna(train_data['ShoppingMall'].median(), inplace=True)
train_data['Spa'].fillna(train_data['Spa'].median(), inplace=True)
train_data['VRDeck'].fillna(train_data['VRDeck'].median(), inplace=True)

test_data['RoomService'].fillna(train_data['RoomService'].median(), inplace=True)
test_data['FoodCourt'].fillna(train_data['FoodCourt'].median(), inplace=True)
test_data['ShoppingMall'].fillna(train_data['ShoppingMall'].median(), inplace=True)
test_data['Spa'].fillna(train_data['Spa'].median(), inplace=True)
test_data['VRDeck'].fillna(train_data['VRDeck'].median(), inplace=True)

train_data.drop("Name", axis=1, inplace=True)
train_data.drop("PassengerId", axis=1, inplace=True)

test_data.drop("Name", axis=1, inplace=True)
test_data.drop("PassengerId", axis=1, inplace=True)


train_data['Cabin_1'] = train_data['Cabin'].str.split('/').str[0]
train_data['Cabin_2'] = train_data['Cabin'].str.split('/').str[1]
train_data['Cabin_3'] = train_data['Cabin'].str.split('/').str[2]
train_data['Cabin_None'] = train_data['Cabin'].apply(lambda x: True if pd.isnull(x) else False)

train_data.drop("Cabin", axis=1, inplace=True)


test_data['Cabin_1'] = test_data['Cabin'].str.split('/').str[0]
test_data['Cabin_2'] = test_data['Cabin'].str.split('/').str[1]
test_data['Cabin_3'] = test_data['Cabin'].str.split('/').str[2]
test_data['Cabin_None'] = test_data['Cabin'].apply(lambda x: True if pd.isnull(x) else False)

test_data.drop("Cabin", axis=1, inplace=True)


train_data['Cabin_2'] = pd.to_numeric(train_data['Cabin_2'], errors='coerce')

#Заполним пропущенные значения в столбце Cabin_2 медианой
train_data['Cabin_2'].fillna(train_data['Cabin_2'].mean(), inplace=True)


test_data['Cabin_2'] = pd.to_numeric(test_data['Cabin_2'], errors='coerce')
test_data['Cabin_2'].fillna(train_data['Cabin_2'].mean(), inplace=True)


train_data = pd.get_dummies(train_data, columns=['HomePlanet', 'Destination', 'Cabin_1', 'Cabin_3'])
test_data = pd.get_dummies(test_data, columns=['HomePlanet', 'Destination', 'Cabin_1', 'Cabin_3'])

train_data.to_csv('data/train_preproc.csv', index=False)
test_data.to_csv('data/test_preproc.csv', index=False)