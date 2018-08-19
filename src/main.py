import sys
import numpy as np

from ClassifierMethods import get_characteristics, get_fft_properties, plot_confussion_matrix
from DataHandler import DataHandler
from ClassifierMethods import ClassifierMethods
from Classifier1 import Classifier1


def function1(dh):
    train_percents = [.5, .5, .5, .5, .5, .5, .5, .5, .5]
    test_percents = [.2, .25, .25, .25, .2, .25, .25, .2, .2]
    class_methods = ClassifierMethods(dh.inputs, train_percents, test_percents)
    get_characteristics(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), np.array([1,1,1]),
                                      mean_=True, stdev_=True )
    #get_fft_properties(class_methods.input_[0][:, 0:-1])
    probs_test = []
    for i in range(0, 9):
        train_idx =  class_methods.train_indices[i]
        test_idx = class_methods.test_indices[i]
        val_idx = class_methods.validation_indices[i]
        X_train = np.array(dh.inputs[i][train_idx[0]: train_idx[1], 0:-1], dtype=float)
        y_train = dh.inputs[i][train_idx[0]: train_idx[1], -1]
        X_test = np.array(dh.inputs[i][test_idx[0]: test_idx[1], 0:-1], dtype=float)
        y_test = dh.inputs[i][test_idx[0]: test_idx[1], -1]
        X_val = np.array(dh.inputs[i][val_idx[0]: val_idx[1], 0:-1], dtype=float)
        y_val = dh.inputs[i][val_idx[0]: val_idx[1], -1]
        clf1 = Classifier1()

        prob, pred_lab = clf1.train_model_and_predict(X_train, y_train, X_test)
        plot_confussion_matrix(pred_lab, y_test, np.unique(y_test), ("logreg_0%d.png" % (i+1)))


if __name__ == '__main__':
    dh = DataHandler(sys.argv[1])
    dh.set_inputs()
    #dh.plot_some_data(0, [0, 500, 1000])
    dh.define_test_train_val()
    dh.select_kBest()
    #print(len(dh.inputs[0][0]))
    #function1(dh)
