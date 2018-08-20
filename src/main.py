import sys
import numpy as np
from DataHandler import DataHandler
from ClassifierMethods import ClassifierMethods


def main(path):
    dh = DataHandler(path)
    dh.set_inputs()
    new_inputs, new_y = dh.radomized_blocks()
    cls_mth = ClassifierMethods(new_inputs, new_y,0.6, 0.2)
    cls_mth.train_and_class()


if __name__ == '__main__':
    main(sys.argv[1])


