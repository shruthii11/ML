import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def gaussian_naive_bayes(data, test_data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X = pd.get_dummies(X)
    nb = GaussianNB()
    nb.fit(X, y)
    y_pred_new = nb.predict(pd.get_dummies(test_data.iloc[:, :-1]))
    accuracy_new = accuracy_score(test_data.iloc[:, -1], y_pred_new)
    print("Accuracy on Test data set:", accuracy_new)

data = pd.read_csv("Prog6_train_data.csv")
test_data = pd.read_csv("Prog6_test_data.csv")
gaussian_naive_bayes(data, test_data)