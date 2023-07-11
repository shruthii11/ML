import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
data = pd.read_csv("Prog4_ID3.csv")
X = data.drop('Play', axis=1)
y = data['Play']
X = pd.get_dummies(X)
y = y.map({'yes': 1, 'no': 0})
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X, y)
export_graphviz(classifier, out_file='tree.dot', feature_names=X.columns)
with open('tree.dot', 'r') as file:
    tree_data = file.read()
print(tree_data)