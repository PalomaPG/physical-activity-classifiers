import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class ClassifierMethods(object):
    def __init__(self, input_, train_percents, test_percents):
        shuffle_narrays_lst(input_)
        self.input_ = input_
        self.test_indices = []
        self.train_indices = []
        self.validation_indices = []
        self.define_sets(train_percents, test_percents)
        self.classifier = None

    def define_sets(self, train_percents, test_percents):

        for i in range(len(self.input_)):
            n_rows = self.input_[i].shape[0]
            train_end = n_rows*train_percents[i]
            test_end = train_end + n_rows * test_percents[i]
            self.train_indices.append((0, train_end))
            self.test_indices.append((train_end, test_end))
            self.validation_indices.append((test_end, -1))

    def plot_roc_curve(self):
        pass


def get_FFT_properties(input_data):
    return []


def get_characteristics(input_data, result_prob,
                        mean_=False, stdev_=False, FFT= False, energy=False):
    chars = []
    chars.append(result_prob)

    if mean_:
        MEAN = np.mean(input_data, axis=1)
        print(MEAN)
        chars.append(MEAN)

    if stdev_:
        STD = np.std(input_data, axis=1)
        print(STD)
        chars.append(STD)

    if FFT:
        chars.append(get_FFT_properties(input_data))

    print(chars)


def plot_confussion_matrix(pred_labels, actual_labels,
                           plot_name, cmap=plt.cm.Blues, show=False):
    classes = ['running', 'treadmill', 'bike', 'climbing', 'walking', 'gymbike',
               'standing', 'jumping', 'descending']
    title = 'Matriz de confusion'
    cm = confusion_matrix(actual_labels, pred_labels)
    cm = cm.astype('float')/ cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
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