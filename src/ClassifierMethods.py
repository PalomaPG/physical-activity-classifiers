import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

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

    def log_reg(self, X_train, y_train, X_test):
        self.logreg_clf.fit(X_train, y_train)
        return self.log_regclf.predict_proba(X_test), self.clf.predict(X_test)

    def rank_features(self):
        for i in range(9):
            X_train = self.input_[i][0: self.train_indices[1], :]
            y_train = (self.labels[0: self.train_indices[1]]).reshape(2700,)
            print('--------------------------------------------')
            print('Select k-best')
            print(self.featSel.kBest_score(X_train, y_train))
            print('Select extratrees')
            print(self.featSel.extree_score(X_train, y_train))
            print('----------------------------------------------')

    def class_rdnforest(self, X_train_score, X_input_score, max_depth, min_samples_split,  random_state,
                        X_train, X_input, y_train, y_input, time_ms, figname='default_rdnfor.png', last=False):
        rdm_for = RandomForestClassifier(max_depth=max_depth, random_state=random_state,
                                        min_samples_split=min_samples_split)
        start = time.process_time()
        rdm_for.fit(X_train, y_train)
        end = time.process_time()
        time_ms = (end - start) + time_ms

        if not last:
            X_train_score = X_train_score + (np.array(rdm_for.predict_proba(X_train)))
            X_input_score = X_input_score + (np.array(rdm_for.predict_proba(X_input)))
            return X_train_score, X_input_score, time_ms
        else:
            y_pred = np.array(rdm_for.predict(X_input))
            y_input.reshape(900, )
            classes = np.unique(y_input)
            plot_confussion_matrix(y_input, y_pred, classes, plot_name=figname,
                                   cmap=plt.cm.Blues, show=False)
            return time_ms

    def class_ada(self, X_train, y_train, X_input, y_input, X_train_score, X_input_score, time_ms, figname='default_ada.png', last=False):
        ada_clf = AdaBoostClassifier(random_state=2, learning_rate=0.1)
        start = time.process_time()
        ada_clf.fit(X_train, y_train)
        end = time.process_time()
        time_ms = (end-start) + time_ms
        if not last:
            X_train_score = X_train_score + (np.array(ada_clf.predict_proba(X_train)))
            X_input_score = X_input_score + (np.array(ada_clf.predict_proba(X_input)))
            return X_train_score, X_input_score, time_ms

        else:
            y_pred = np.array(ada_clf.predict(X_input))
            y_input.reshape(900, )
            classes = np.unique(y_input)
            plot_confussion_matrix(y_input, y_pred, classes, plot_name=figname,
                                   cmap=plt.cm.Blues, show=False)
            return time_ms

    def train_and_class(self, test=False):

        rdn_for = np.zeros((900, 9))
        ada_score = np.zeros((900, 9))
        rdn_train = np.zeros((2700, 9))
        ada_train_score = np.zeros((2700, 9))
        time_rdfor = 0
        time_ada = 0

        for i in range(9):
            X_train = self.input_[i][0: self.train_indices[1], :]
            y_train = (self.labels[0: self.train_indices[1]]).reshape(2700,)

            if not test:
                X_input = self.input_[i][self.validation_indices[0]:, :]
                y_input = (self.labels[self.validation_indices[0]: ]).reshape(900,)
            else:
                X_input = self.input_[i][self.test_indices[0]: self.test_indices[1], :]
                y_input = (self.labels[self.test_indices[0]:self.test_indices[1]]).reshape(900,)
            rdn_train, rdn_for, time_rdfor = self.class_rdnforest(time_ms=time_rdfor, X_input_score=rdn_for, X_train_score=rdn_train, max_depth=4,
                                                      min_samples_split=2, random_state=2, X_train=X_train,
                                                      X_input=X_input, y_train=y_train, y_input=y_input)
            ada_train_score, ada_score, time_ada = self.class_ada(time_ms=time_ada, X_train=X_train, X_input=X_input, X_train_score=ada_train_score,
                                                        X_input_score=ada_score, y_input=y_input, y_train=y_train)

        time_rdfor = self.class_rdnforest(time_ms=time_rdfor, X_train_score=None, X_input_score=None, max_depth=4, min_samples_split=2, random_state=2,
                             X_train=rdn_train, X_input=rdn_for, y_train=y_train, y_input=y_input, figname='rdfor_class.png', last=True)
        time_ada= self.class_ada(time_ms=time_ada, X_train_score=None, X_input_score=None, X_train=X_train,
                       X_input=X_input, y_train=y_train, y_input=y_input, figname='ada_class.png', last=True)
        print('Training time for Random Forest: %f' % time_rdfor)
        print('Training time for AdaBoost: %f' % time_ada)


    def train_and_class_selfeat(self, kbest = True, test=False):

        rdn_for = np.zeros((900, 9))
        ada_score = np.zeros((900, 9))
        rdn_train = np.zeros((2700, 9))
        ada_train_score = np.zeros((2700, 9))
        time_rdfor = 0
        time_ada = 0

        for i in range(9):
            X_train = self.input_[i][0: self.train_indices[1], :]
            y_train = (self.labels[0: self.train_indices[1]]).reshape(2700,)
            if not test:
                X_input = self.input_[i][self.validation_indices[0]:, :]
                y_input = (self.labels[self.validation_indices[0]: ]).reshape(900,)
            else:
                X_input = self.input_[i][self.test_indices[0]: self.test_indices[1], :]
                y_input = (self.labels[self.test_indices[0]:self.test_indices[1]]).reshape(900,)

            if kbest:
                print('Seleccionando k-mejores... K-best')
                X_train, X_input = self.featSel.kBest_fit(X_train=X_train, X_input=X_input)
            else:
                print('Seleccionando por extra trees...')
                X_train, X_input = self.featSel.extree_fit(X_train=X_train, X_input=X_input)

            rdn_train, rdn_for, time_rdfor = self.class_rdnforest(time_ms=time_rdfor, X_input_score=rdn_for, X_train_score=rdn_train, max_depth=4,
                                                      min_samples_split=2, random_state=2, X_train=X_train,
                                                      X_input=X_input, y_train=y_train, y_input=y_input)
            ada_train_score, ada_score, time_ada = self.class_ada(time_ms=time_ada, X_train=X_train, X_input=X_input, X_train_score=ada_train_score,
                                                        X_input_score=ada_score, y_input=y_input, y_train=y_train)

        time_rdfor = self.class_rdnforest(time_ms=time_rdfor, X_train_score=None, X_input_score=None, max_depth=4, min_samples_split=2, random_state=2,
                             X_train=rdn_train, X_input=rdn_for, y_train=y_train, y_input=y_input, figname=['kbestfeat_rdfor.png' if kbest else 'extree_rdfor.png' ][0], last=True)
        time_ada= self.class_ada(time_ms=time_ada, X_train_score=None, X_input_score=None, X_train=X_train,
                       X_input=X_input, y_train=y_train, y_input=y_input, figname=['kbestfeat_ada.png' if kbest else 'extree_ada.png' ][0], last=True)
        print('Training time for Random Forest: %f' % time_rdfor)
        print('Training time for AdaBoost: %f' % time_ada)


def plot_confussion_matrix(pred_labels, actual_labels, classes,
                           plot_name, cmap=plt.cm.Blues, show=False):

    title = 'Matriz de confusion'
    cm = confusion_matrix(actual_labels, pred_labels)
    print(classification_report(actual_labels, pred_labels, labels=classes))
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