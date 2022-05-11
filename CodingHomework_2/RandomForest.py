import numpy as np
import random

class DataLoader():
    def __init__(self):
        self.file_name = './Employee.csv'
        self.load_data()

    def load_data(self):
        self.data = []
        f = open(self.file_name, 'r')
        # parse csv
        data = [[item for item in line.strip().split(',')] for line in f.readlines()]
        f.close()
        self.title = data[0]
        self.data = data[1:] # merge
    
class DecisionTree(DataLoader):
    def __init__(self):
        super().__init__()
    
    
class RandomForest(DataLoader):
    def __init__(self):
        super().__init__()
        self.train_ratio = 0.8
        # map non numerical features to int
        self.education = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
        self.city = {'Bangalore': 0, 'Pune': 1, 'New Delhi': 2}
        self.gender = {'Male': 0, 'Female': 1}
        self.everbenched = {'No': 0, 'Yes': 1}
        self.prepare_data()
        # self.train()
        
    def prepare_data(self):
        # map non numerical features into numbers
        for line in self.data:
            line[0] = self.education[line[0]]
            line[2] = self.city[line[2]]
            line[5] = self.gender[line[5]]
            line[6] = self.everbenched[line[6]]
        self.x_data = np.array([line[:-1] for line in self.data], dtype = 'int')
        self.x_data = np.insert(self.x_data, 0, np.ones(self.x_data.shape[0]), axis = 1) # add bias feature 1
        
        self.y_data = np.array([[line[-1]] for line in self.data], dtype = 'int8')
    
        # save all possible values of each feature
        self.features = []
        for i in range(self.x_data.shape[1]):
            x_single = self.x_data[:, i]
            self.features.append(list(set(x_single)))
        
    def bootstrap(self): 
        self.total = self.x_data.shape[0]
        self.N = int(self.train_ratio * self.total)
        # bootstrap train set
        self.train_index = np.random.randint(0, self.total, self.N)
        self.x_train = self.x_data[self.train_index]
        self.y_train = self.y_data[self.train_index]
        # else are test set
        self.test_index = list(set(range(self.total)) - set(self.train_index))
        self.x_test = self.x_data[self.test_index]
        self.y_test = self.y_data[self.test_index]

    def entrophy(self, x):
        if x == 0 or x == 1:
            return 0
        return - x * np.log(x) - (1 - x) * np.log(1 - x)

    def decision_tree(self, x, y, feature_index):
        n_true = y[y == 1].shape[0]
        hd = self.entrophy(n_true / self.N)

        # pick a feature each step
        gd_max = 0
        index = feature_index[0] # which index to move
        for i in feature_index: # find feature with max gd
            values = self.features[i]
            xi = x[:, i] # picked row
            hi = 0
            for v in values:
                yi = y[xi == v]
                n_true = yi[yi == 1].shape[0]
                if yi.shape[0] != 0:
                    hi += yi.shape[0] / self.N * self.entrophy(n_true / yi.shape[0])
            gd = hd - hi
            if gd > gd_max:
                gd_max = gd
                index = i
        feature_index.remove(index) # next step's feature index
        values = self.features[index]
        decisions = {} # map: value -> prediction
        xi = x[:, index]
        for v in values:
            xv = x[xi == v, :]
            yv = y[xi == v]
            if xv.shape[0] == 0:
                continue
            if feature_index == []:
                decisions[v] = (yv[yv == 1].shape[0] > yv[yv == 0].shape[0])
            elif yv[yv == 1].shape[0] == yv.shape[0]:
                decisions[v] = True
            elif yv[yv == 0].shape[0] == yv.shape[0]:
                decisions[v] = False
            else: # not leaf node
                decisions[v] = self.decision_tree(xv, yv, feature_index)
        return index, decisions
    
    def train(self):
        # params
        self.M = self.x_data.shape[1] - 1 # number of features
        self.m = int(np.log2(self.M) + 1)
        self.B = 10 # number of trees

        # build B trees
        self.forest = []
        for i in range(self.B):
            self.bootstrap()
            # random pick m features
            feature_index = random.sample(range(1, self.M + 1), self.m)
            
            # decision tree
            self.forest.append(self.decision_tree(self.x_train, self.y_train, feature_index))
            # print(self.forest[i])
        return self.forest
    
    def make_decision(self, tree, x):
        index, value_dict = tree
        xi = x[:, index]
        predict = np.array([-1 for _ in range(len(xi))])
        for v in value_dict.keys():
            items = (xi == v)
            xv = x[xi == v, :]
            if isinstance(value_dict[v], tuple):
                predict[items] = self.make_decision(value_dict[v], xv)
            else:
                predict[items] = 1 if value_dict[v] else 0
        return predict
    
    def test(self):
        self.predict = []
        for tree in self.forest:
            self.predict.append(self.make_decision(tree, self.x_test).tolist())
        self.predict = np.array(self.predict).transpose(1, 0) # (N, M)
        true_num = np.sum(self.predict, axis = 1, keepdims = True)
        predict = true_num > (self.predict.shape[1] // 2)
        predict.reshape(predict.shape[1], predict.shape[0])
        
        accuracy = np.sum(predict == self.y_test) / self.y_test.shape[0]
        print('number = ' + str(self.y_test.shape[0]))
        print('accuracy = ' + str(accuracy))


if __name__ == '__main__':
    forest = RandomForest()
    forest.train()
    forest.test()