import numpy as np

x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
y = np.array([4, 2.4, 1.5, 1.0, 1.2, 1.5,  1.8, 2.6, 3.0, 4.0, 4.5, 5.0,  6.0])

def find_pivot(x, y):
    min_loss = 0x3fffffff
    for j in x:
        y1 = y[x <= j]
        y2 = y[x > j]
        g1 = np.sum(y1) / y1.shape[0]
        g2 = 0 if y2.shape[0] == 0 else np.sum(y2) / y2.shape[0]
        loss1 = np.sum((y1 - g1)**2)
        loss2 = np.sum((y2 - g2)**2)
        loss = loss1 + loss2
        # print('j = {}, loss = {}'.format(j, loss))
        if loss < min_loss:
            min_loss = loss
            pivot = j
            epsilon1 = np.sqrt(loss1 / y1.shape[0])
            epsilon2 = 0 if y2.shape[0] == 0 else loss2 / y2.shape[0]
    return pivot, epsilon1, epsilon2

def CART_loop(x, y):
    epsilon0 = 0.1
    pivot, epsilon1, epsilon2 = find_pivot(x, y)
    # print('j = {}, epsilon1 = {}, epsilon2 = {}'.format(pivot, epsilon1, epsilon2))
    pivots = [pivot]
    x1 = x[x <= pivot]
    x2 = x[x > pivot]
    y1 = y[x <= pivot]
    y2 = y[x > pivot]

    if epsilon1 > epsilon0:
        pivot1 = CART_loop(x1, y1)
        pivots = pivot1 + pivots

    if epsilon2 > epsilon0:
        pivot2 = CART_loop(x2, y2)
        pivots = pivots + pivot2
    
    return pivots

def CART_Regression(x, y, x0):
    pivots = CART_loop(x, y)
    print('pivot = {}'.format(pivots))
    for j in pivots:
        yi = y[x <= j]
        #print('j = {}, g = {}'.format(j, np.sum(yi) / yi.shape[0]))
        if x0 <= j:
            y0 = np.sum(yi) / yi.shape[0]
            break
        y = y[x > j]
        x = x[x > j]
        #print(x)
    if x0 > pivots[-1]:
        y0 = y[-1] if y.shape[0] == 0 else np.sum(y) / y.shape[0]
    print('x = {} y = {}'.format(x0, y0))

if __name__ == '__main__':
    CART_Regression(x, y, 0.76)
    #find_pivot(x, y)