import numpy as np
import sys
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self):
        self.file_list = ['./Dataset_traffic-driving-style-road-surface-condition/opel_corsa_01.csv', 
                './Dataset_traffic-driving-style-road-surface-condition/opel_corsa_02.csv', 
                './Dataset_traffic-driving-style-road-surface-condition/peugeot_207_01.csv', 
                './Dataset_traffic-driving-style-road-surface-condition/peugeot_207_02.csv'
                ]
        self.load_data()

    def load_data(self):
        self.data = []
        for file_name in self.file_list:
            f = open(file_name, 'r')
            # parse csv
            data = [[item if item != '' else '0' for item in line.strip().replace(',', '.').split(';')] for line in f.readlines()]
            f.close()
            self.title = data[0]
            self.data += data[1:] # merge
    
class GaussionGeneration(DataLoader):
    def __init__(self):
        super().__init__()
        self.train_ratio = 0.8
        self.labels = {'EvenPaceStyle': 0, 'AggressiveStyle': 1}
        self.prepare_data()
        
    def prepare_data(self):
        self.x_data = np.array([line[:-3] for line in self.data], dtype = 'float')
        self.y_data = np.array([self.labels[line[-1]] for line in self.data])
        self.y_data = np.eye(2)[self.y_data] # onehot
    
        # scale
        x_max = np.max(self.x_data, axis = 0)
        x_min = np.min(self.x_data, axis = 0)
        self.x_data = (self.x_data - x_min) / x_max
        
        # random divide into train and test set
        self.total = self.x_data.shape[0]
        self.N = int(self.train_ratio * self.total)
        self.data_index = np.array(range(self.total))
        np.random.shuffle(self.data_index)
        self.x_train = self.x_data[self.data_index[:self.N]]
        self.y_train = self.y_data[self.data_index[:self.N]]
        self.x_test = self.x_data[self.data_index[self.N:]]
        self.y_test = self.y_data[self.data_index[self.N:]]

    def train(self):
        # params
        self.N = self.x_train.shape[0] # number of samples
        self.M = self.x_train.shape[1] # number of features
        self.K = 3 # number of categories

        # gaussion generation model
        self.pc1 = np.sum(self.y_train[:, 1]) / self.N
        self.pc0 = np.sum(self.y_train[:, 0]) / self.N
        self.miu1 = np.dot(self.x_train.T, self.y_train[:, 1]) / np.sum(self.y_train[:, 1])
        self.miu0 = np.dot(self.x_train.T, self.y_train[:, 0]) / np.sum(self.y_train[:, 0])
        self.miu = np.dot(self.y_train[:, 1].reshape(self.N, 1), self.miu1.reshape(1, self.M)) + np.dot(self.y_train[:, 0].reshape(self.N, 1), self.miu0.reshape(1, self.M))
        self.bias = self.x_train - self.miu
        self.conv = np.dot(self.bias.T, self.bias) / self.N
        self.inv = np.linalg.pinv(self.conv) # inverse of conv matrix
        self.a = 1 / (np.sqrt(2 * np.pi) * (np.linalg.norm(self.conv)) ** (self.M / 2))
        
    
    def test(self):
        number = self.x_test.shape[0]
        x_predict = []
        for x in self.x_test:
            px_c1 = self.a * np.exp(-(x - self.miu1).reshape(1, self.M) @ self.inv @ (x - self.miu1).reshape(self.M, 1))
            px_c0 = self.a * np.exp(-(x - self.miu0).reshape(1, self.M) @ self.inv @ (x - self.miu0).reshape(self.M, 1))
            pxc1 = px_c1 * self.pc1
            pxc0 = px_c0 * self.pc0
            pc1_x = pxc1 / (pxc1 + pxc0) # p(c1|x) = p(x|c1)p(c1) / (p(x|c1)p(c1) + p(x|c0)p(c0))
            pc0_x = pxc0 / (pxc1 + pxc0)
            x_predict.append(int(pc1_x > pc0_x)) # is AgressiveStyle or not
        x_predict = np.array(x_predict)
        # sum results
        tp = np.sum(np.multiply(self.y_test[:, 1], x_predict))
        fp = np.sum(np.multiply(self.y_test[:, 0], x_predict))
        tn = np.sum(np.multiply(self.y_test[:, 0], 1 - x_predict))
        fn = np.sum(np.multiply(self.y_test[:, 1], 1 - x_predict))
        print('tp = {} fp = {} tn = {} fn = {}'.format(tp, fp, tn, fn))
        accuracy = (tp + tn) / number
        print('accuracy = {}'.format(accuracy))
        
        
if __name__ == '__main__':
    np.seterr(divide='ignore',invalid='ignore')
    driving_style = GaussionGeneration()
    driving_style.train()
    driving_style.test()