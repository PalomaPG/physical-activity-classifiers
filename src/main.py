import sys
import numpy as np

from DataHandler import DataHandler

def main(path):
    dh = DataHandler(sys.argv[1])
    dh.set_inputs()
    new_inputs = dh.radomized_blocks()
    lim_train = int(4500*0.6)
    lim_test = int(4500*0.6) + int(4500*0.2)


if __name__ == '__main__':
    main(sys.argv[1])


