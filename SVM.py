from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

report = classification_report(y_test, y_pred)
print("Classification report:\n", report)

clf = SVC(C=100,gamma=0.0001)
pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(X_train)
clf.fit(X_train2, y_train)

plot_decision_regions(X_train2, y_train, clf=clf, legend=2)
plt.xlabel('mean radius')
plt.ylabel('mean texture')
plt.title('SVM decision boundary')
plt.show()