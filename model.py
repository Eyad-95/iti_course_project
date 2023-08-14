# import pandas as pd
# import numpy as np

# df=pd.read_csv('BankNote_Authentication.csv')

# X=df.iloc[:,:-1]
# y=df.iloc[:,-1]

# from sklearn.model_selection import train_test_split

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# from sklearn.ensemble import RandomForestClassifier
# classifier=RandomForestClassifier()
# classifier.fit(X_train,y_train)

# y_pred=classifier.predict(X_test)

# from sklearn.metrics import accuracy_score
# score=accuracy_score(y_test,y_pred)

# import pickle
# pickle_out = open("classifier.pkl","wb")
# pickle.dump(classifier, pickle_out)
# pickle_out.close()


import pickle
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Lasso

url = 'train_reg.csv'
df = pd.read_csv(url)

# Import Lasso and matplotlib
# Instantiate a lasso regression model
lasso = Lasso(5)
X = df.drop(columns=["cnt", "dteday"])  # Drop the "cnt" column
y = df["cnt"]

# Fit the model to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
column_names_array = df.columns.to_numpy()

# Column names to remove
columns_to_remove = ["cnt", "dteday"]

# Remove specified columns from the array
filtered_column_names = [
    col for col in column_names_array if col not in columns_to_remove]

X = df[["registered", "casual"]]


parameter_grid = {
    "Linear Regression": {},
    "Ridge Regression": {"alpha": [0.1, 1.0, 10.0]},
    "Lasso Regression": {"alpha": [0.1, 1.0, 10.0]},
    "Decision Tree Regression": {"max_depth": [None, 10, 20],
                                 "min_samples_split": [2, 5, 10]},
    "Random Forest Regression": {"n_estimators": [50, 100, 200],
                                 "max_depth": [None, 10, 20],
                                 "min_samples_split": [2, 5, 10]},
    "Gradient Boosting Regression": {"n_estimators": [50, 100, 200],
                                     "max_depth": [3, 4, 5],
                                     "learning_rate": [0.01, 0.1, 0.2]},
    "Support Vector Regression": {"C": [0.1, 1.0, 10.0],
                                  "kernel": ["linear", "rbf"]},
    "K-Nearest Neighbors Regression": {"n_neighbors": [3, 5, 7, 10],
                                       "weights": ["uniform", "distance"]}
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

import json

filename = "data.json"

models_to_be_printed = {}

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(),
    "Gradient Boosting Regression": GradientBoostingRegressor(),
    "Support Vector Regression": SVR(),
    "K-Nearest Neighbors Regression": KNeighborsRegressor()
}


results = []
# Loop through the models' values
for name, model in models.items():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    grid_search = GridSearchCV(model, parameter_grid[name], cv=kf)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)
    models_to_be_printed[name] = {
        'best_params': best_params,
        'best_score': best_score
    }

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    r_squared = best_model.score(X_test, y_test)

    results.append(r_squared)

    models.keys()

    x_label = ['Linear\nRegression', 'Ridge\nRegression', 'Lasso\nRegression', 'Decision\nTree/nRegression', 'Random\nForest\nRegression',
               'Gradient\nBoosting\nRegression', 'Support\nVector\nRegression', 'K-Nearest\nNeighbors\nRegression']

with open(filename, "w") as json_file:
    json.dump(models_to_be_printed, json_file)

model = LinearRegression()
model.fit(X, y)

pickle_out = open("classifier.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
