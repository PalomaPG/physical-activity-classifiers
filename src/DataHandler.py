import numpy as np


class DataHandler(object):

    def __init__(self, path_):
        self.path = path_
        self.inputs = []
        self.labels = ['running', 'treadmill', 'bike',
                       'climbing', 'walking', 'gymbike',
                       'standing', 'jumping', 'descending']

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
                    np_array = np.loadtxt(fname, delimiter=',', dtype=object)
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

