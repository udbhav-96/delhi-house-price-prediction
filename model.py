import pandas as pd
import pickle
import numpy as np
df = pd.read_csv("assets/csv/Delhi_for_model.csv")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X = df[['area', 'latitude', 'longitude', 'Bedrooms', 'Bathrooms', "Status", "neworold", "type_of_building"]].values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
my_pred = model.predict(X_test)
#my_pred = model.predict([np.array([1100, 28.608850, 77.460560, 2, 2, 1, 1, 1])])
#pickle.dump(model, open("delhi_trained_model.pkl", "wb"))
#print(model.score(X_train, y_train))
#print(model.score(X_test, y_test))
print(my_pred)