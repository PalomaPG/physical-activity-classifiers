import sys
from DataHandler import DataHandler
from ClassifierMethods import ClassifierMethods


def main(path):
    dh = DataHandler(path)
    dh.set_inputs()
    new_inputs, new_y = dh.radomized_blocks()
    cls_mth = ClassifierMethods(new_inputs, new_y, 0.6, 0.2)
    #cls_mth.train_and_class()
    cls_mth.rank_features()
    #cls_mth.train_and_class_selfeat()
    #cls_mth.train_and_class_selfeat(kbest=False)
    cls_mth.train_and_class(test=True)
    cls_mth.train_and_class_selfeat(test=True)

if __name__ == '__main__':
    main(sys.argv[1])


