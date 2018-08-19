from sklearn.linear_model import LogisticRegression
from sklearn import metrics
class Classifier1():

    def __init__(self):
        self.clf = LogisticRegression()

    def train_model_and_predict(self, X_train, y_train,
                                X_test):
        self.clf.fit(X_train, y_train)
        return self.clf.predict_proba(X_test), self.clf.predict(X_test)