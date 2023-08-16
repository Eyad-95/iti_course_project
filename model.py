import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

url = 'neo.csv'
df = pd.read_csv(url)


X = df[['relative_velocity', 'miss_distance']].values
y = df['hazardous'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X = scaler.fit_transform(X)

kf = KFold(n_splits=6, random_state=42, shuffle=True)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

pickle_out = open("classifier.pkl", "wb")
pickle.dump({"model": classifier, "cm": cm,"scaler":scaler}, pickle_out)
pickle_out.close()
