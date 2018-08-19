import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

import pandas as pd

class DataHandler(object):

    def __init__(self, path_):
        self.path = path_
        self.inputs = []
        self.labels = ['bike', 'climbing', 'descending',
                        'gymbike',  'jumping', 'running',
                        'standing',  'treadmill', 'walking']
        self.new_inputs = []
        self.new_y = None

    def set_inputs(self):
        for i in range(1, 10):
            self.inputs.append(self.look_at_sensor(i))

    def look_at_sensor(self, n_sensor):

        specific_path = self.path+('S0%d/' % n_sensor)
        ext = '.csv'
        result = []
        for act in self.labels:
            act_result = []
            labels = [[act]]
            files = [specific_path + act + str(i) + ext for i in range(1, 6)]
            try:
                for fname in files:
                    np_array = np.loadtxt(fname, delimiter=',', dtype=float)
                    if len(labels) == 1:
                        labels = np.array(labels * np_array.shape[0], dtype=object)
                    act_result.append(np_array)
                act_result = np.concatenate(act_result, axis=1)
                act_result = np.append(act_result, labels, axis=1)
                result.append(act_result)
            except OSError:
                pass
        result = np.concatenate(result)
        return result

    def plot_some_data(self, block, rows_ids):

        plt.style.use('fivethirtyeight')
        fig, ax = plt.subplots()

        for i in range(0, len(rows_ids)):
            y = self.inputs[block][rows_ids[i]][:-1]
            x = np.arange(0, len(y))
            label = self.inputs[block][rows_ids[i]][-1:][0]
            ax.plot(x, self.inputs[block][rows_ids[i]][:-1], label=label)
        ax.set_title('Sample')
        ax.legend(title='Activity', loc='best')
        plt.show()

    def calc_macroclass_feat(self, block):
        print(block)
        X = self.inputs[block][:, :-1]
        y = self.inputs[block][:, -1]

        n_rows = X.shape[0]
        y = y.reshape((n_rows, 1))
        macro_feat = []
        #RMS
        rms = np.sum(np.power(X, 2.0) / 45, axis=1)
        rms = np.float64(rms)

        macro_feat.append(np.mean(X, axis=1).reshape(1, n_rows))# Mean
        macro_feat.append(np.var(X, axis=1).reshape(1, n_rows))# Variance
        macro_feat.append(np.min(X, axis=1).reshape(1, n_rows))# Minimum
        macro_feat.append(np.max(X, axis=1).reshape(1, n_rows))# Maximum
        macro_feat.append(np.ptp(X, axis=1).reshape(1, n_rows))# Range

        zero_crosses = np.diff(X > 0, axis=1)
        macro_feat.append((np.sum(zero_crosses, axis=1)/45.0).reshape(1, n_rows)) #Crossing rate
        macro_feat.append(np.sqrt(rms).reshape(1, n_rows))  # RMS
        macro_feat.append(sts.skew(X.astype(float), axis=1).reshape(1, n_rows))  # Skew
        macro_feat.append(np.apply_along_axis(get_entropy, 1, X).reshape(1, n_rows))
        macro_feat.append(sts.kurtosis(X, axis=1).reshape(1, n_rows))  # Kurtosis de Fisher
        
        X = np.concatenate(macro_feat).T

        if n_rows < 4500 :
            print('this is the case')
            X, y = self.fill_absent_data(X, y)
        print(type(X))

        return X, y

    def fill_absent_data(self, X, y):
        n_att = (X.shape)[1]
        labels_y = np.unique(y)
        new_X = np.zeros((4500, n_att))
        new_y = np.chararray((4500, 1), itemsize=15, unicode=True)
        i = 0 # across new_X
        j = 0 # across X
        for l in self.labels:
            if l not in labels_y:
                new_X[i : i+500, :] = -9999
                new_y[i : i+500] = l
                i = i + 500
            else:
                new_X[i:i+500, :] = X[j:j+500, :]
                new_y[i:i+500] = l
                i = i + 500
                j = j + 500

        return new_X, new_y

    def radomized_blocks(self):
        indices = np.arange(0, 4500)
        np.random.shuffle(indices)
        for i in range(9):
            X, y = self.calc_macroclass_feat(i)
            X = X[indices, :]
            if i == 8:
                y = y[indices]
                self.new_y = y
            self.new_inputs.append(X)
        return self.new_inputs

    def concatenate_attrs(self):
        block_lst = []
        for i in range(9):
            X, y = self.calc_macroclass_feat(i)
            block_lst.append(X)
        new_X = np.concatenate(block_lst, axis=1)
        return new_X, y

    def define_test_train_val(self):
        X,y = self.concatenate_attrs()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=7)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=.5, random_state=41)
        return X_train, X_test, X_val, y_train, y_test, y_val

def get_entropy(row):
    h = sts.entropy(pd.Series(row).value_counts())/45.0
    return h