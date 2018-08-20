import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from FeatureSelector import FeatureSelector

class ClassifierMethods(object):

    def __init__(self, input_, labels, train_percent, test_percent):
        self.input_ = input_
        self.labels = labels
        self.define_indices(train_percent, test_percent)
        self.featSel = FeatureSelector()
        self.logreg_clf = LogisticRegression()


    def define_indices(self, train_percent, test_percent):

        n_rows = 4500#input_[i].shape[0]
        train_end = int(n_rows*train_percent)
        test_end = int(train_end + n_rows * test_percent)
        self.train_indices = (0, train_end)
        self.test_indices = (train_end, test_end)
        self.validation_indices = (test_end, -1)

    def plot_roc_curve(self):
        pass

    def log_reg(self, X_train, y_train,
                                X_test):
        self.logreg_clf.fit(X_train, y_train)
        return self.log_regclf.predict_proba(X_test), self.clf.predict(X_test)

    def rank_features(self):
        for i in range(9):
            X_train = self.input_[i][0: self.train_indices[1], :]
            y_train = self.input_[i][0: self.train_indices[1]]
            print(self.featSel.kBest_score(X_train, y_train))
            print(self.featSel.extree_score(X_train, y_train))

    def class_rdnforest(self, new_input_test, new_input_train, max_depth, min_samples_split,  random_state,
                        X_train, X_val, y_train, y_val, block, test=False):
        rdm_for = RandomForestClassifier(max_depth=max_depth, random_state=random_state,
                                        min_samples_split=min_samples_split)
        rdm_for.fit(X_train, y_train.reshape(2700,))
        if not test:
            new_input_train = new_input_train + (np.array(rdm_for.predict_proba(X_train)))
            new_input_test = new_input_test + (np.array(rdm_for.predict_proba(X_val)))
            return new_input_test, new_input_train
        else:
            y_pred = np.array(rdm_for.predict(X_val))
            y_val.reshape(900, )
            classes = np.unique(y_val)
            plot_confussion_matrix(y_val, y_pred, classes, plot_name=('rnd_for_%d.png' % block),
                                   cmap=plt.cm.Blues, show=False)




    def train_and_class(self):

        rdn_for = np.zeros((900, 9))
        rdn_train = np.zeros((2700, 9))
        for i in range(9):
            X_train = self.input_[i][0: self.train_indices[1], :]
            y_train = self.labels[0: self.train_indices[1]]
            X_val = self.input_[i][self.validation_indices[0]:, :]
            y_val = self.labels[self.validation_indices[0]: ]
            rdn_for, rdn_train=self.class_rdnforest(rdn_for, rdn_train, max_depth=4, min_samples_split=2, random_state=2,
                                 X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, block=i)

        self.class_rdnforest(None, None, max_depth=4, min_samples_split=2, random_state=2,
                             X_train=rdn_train, X_val=rdn_for, y_train=y_train, y_val=y_val, block=11, test=True)

def plot_confussion_matrix(pred_labels, actual_labels, classes,
                           plot_name, cmap=plt.cm.Blues, show=False):

    title = 'Matriz de confusion'
    cm = confusion_matrix(actual_labels, pred_labels)
    cm = cm.astype('float')/ cm.sum(axis=1)[:, np.newaxis]
    fmt = '.2f'
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if show:
        plt.show()
    plt.savefig(plot_name)


def shuffle_narrays_lst(input_):
    for i in range(len(input_)):
        np.random.shuffle(input_[i])