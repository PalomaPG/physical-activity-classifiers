import sys
import numpy as np

from DataHandler import DataHandler
from ClassifierMethods import ClassifierMethods


if __name__ == '__main__':
    dh = DataHandler(sys.argv[1])
    dh.set_inputs()
    train_percents = [.5, .5, .5, .5, .5, .5, .5, .5, .5]
    test_percents = [.2, .25, .25, .25, .2, .25, .25, .2, .2]
    class_methods = ClassifierMethods(dh.inputs, train_percents, test_percents)
    get_characteristics(np.array([[1,1,1],[2,2,2],[3,3,3]]), np.array([1,1,1]),
                                      mean_=True, stdev_=True )

