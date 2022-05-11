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
    
class RoadSurfaceCondition(DataLoader):
    def __init__(self):
        super().__init__()
        self.train_ratio = 0.8
        self.labels = {'SmoothCondition': 0, 'FullOfHolesCondition': 1, 'UnevenCondition': 2}
        self.prepare_data()
        
    def prepare_data(self):
        self.x_data = np.array([line[:-3] for line in self.data], dtype = 'float')
        self.x_data = np.insert(self.x_data, 0, np.ones(self.x_data.shape[0]), axis = 1) # add bias feature 1
        
        self.y_data = np.array([self.labels[line[-3]] for line in self.data])
        self.y_data = np.eye(3)[self.y_data] # onehot
    
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
        self.w = np.random.rand(self.K, self.M)
        self.epoch = 0
        self.learning_rate = 0.0001
        self.threshold = 0.1
        self.max_epoch = 2000

        epoch = self.epoch
        alpha = self.learning_rate
        loss = 0x3fffffff
        w = self.w
        loss_list = []
        while loss > self.threshold and epoch < self.max_epoch:
            # SGD
            #index = np.random.randint(0, self.N)
            #x = self.x_train[index].reshape(1, self.M)
            #y = self.y_train[index].reshape(1, self.K)
            # GD
            x = self.x_train
            y = self.y_train
            # forwarding
            a = np.dot(x, w.T) # (N, M) * (M, K) = (N, K)
            t = np.exp(a) / np.sum(np.exp(a), axis = 1, keepdims = True) # softmax
            loss = -np.sum(np.multiply(y, np.log(t))) # cross entrophy
            loss_list.append(loss)
            epoch += 1
            if epoch >= 100 and epoch % 10 == 0:
                alpha = 0.95 * alpha # reduce learning rate
            sys.stderr.write('[Epoch {}] loss = {}\n' .format(epoch, loss))
            # backwarding
            dw = np.dot((t - y).T, x)
            w = w - alpha * dw
        self.w = w
        for i in w:
            sys.stderr.write(' '.join(map(str, i)) + '\n')
        # plot results
        plt.figure()
        plt.title('LogisticRegression')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(loss_list)
        plt.savefig('LogisticRegression.png')
    
    def test(self):
        self.t = np.dot(self.x_test, self.w.T) # prediction
        self.y_predict = np.argmax(self.t, axis = 1) # predict category
        self.y_truth = np.argmax(self.y_test, axis = 1) # ground truth category
        accuracy = np.sum(self.y_truth == self.y_predict) / self.y_test.shape[0]
        print('number = ' + str(self.y_test.shape[0]))
        print('accuracy = ' + str(accuracy))


if __name__ == '__main__':
    road = RoadSurfaceCondition()
    road.train()
    road.test()