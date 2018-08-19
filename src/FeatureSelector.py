import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

class FeatureSelector(object):

    def __init__(self):
        self.sel_kbest = SelectKBest(f_classif, k=5)
        self.extree_clf = ExtraTreesClassifier()

    def kBest_score(self, X_train, y_train):
        self.sel_kbest.fit(X_train, y_train)
        return np.argsort(self.sel_kbest.scores_)

    def kBest_modify_data(self, X_train, X_test, X_val):
        return self.sel_kbest.transform(X_train), self.sel_kbest.transform(X_test), self.sel_kbest.transform(X_val)

    def extree_score(self, X_train, y_train):
        self.extree_clf.fit(X_train, y_train)
        return np.argsort(self.extree_clf.feature_importances_)

    def extree_fit(self, X_train, X_test, X_val):
        model = SelectFromModel(self.extree_clf, prefit=True)
        return model.transform(X_train), model.transform(X_test), model.transform(X_val)

