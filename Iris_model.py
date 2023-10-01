import pickle 

import pandas  as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.csv')

x = df.iloc[:, 0:4]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

score = accuracy_score(y_test, y_pred)

pickle_out = open("Iris_model.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()