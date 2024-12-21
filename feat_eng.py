#Создадим новые фичи
import pandas as pd
train_data = pd.read_csv('data/train_transformed.csv')
test_data = pd.read_csv('data/test_transformed.csv')

train_data['TotalSpent'] = train_data['RoomService'] + train_data['FoodCourt'] + train_data['ShoppingMall'] + train_data['Spa'] + train_data['VRDeck']

#Добавим столбцец с группами по Age
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['0-18', '18-30', '30-45', '45-60', '60-100'])

train_data = pd.get_dummies(train_data, columns=['AgeGroup'])



test_data['TotalSpent'] = test_data['RoomService'] + test_data['FoodCourt'] + test_data['ShoppingMall'] + test_data['Spa'] + test_data['VRDeck']
test_data['AgeGroup'] = pd.cut(test_data['Age'], bins=[0, 18, 30, 45, 60, 100], labels=['0-18', '18-30', '30-45', '45-60', '60-100'])
test_data = pd.get_dummies(test_data, columns=['AgeGroup'])


train_data.to_csv('data/train_feateng.csv', index=False)
test_data.to_csv('data/test_feateng.csv', index=False)