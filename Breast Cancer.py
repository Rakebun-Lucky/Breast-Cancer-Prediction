import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()

#print(cancer.feature_names)
#print(cancer.target_names)

x = cancer.data
y = cancer.target

best_accuracy = 0
for i in range(10):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
    #print(x_train, y_train)
    classes = ['malignant' 'benign']
    clf = svm.SVC(kernel = 'linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    if accuracy>best_accuracy:
        best_accuracy = accuracy

print(best_accuracy)
