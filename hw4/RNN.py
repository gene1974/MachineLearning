import numpy as np

b = np.array([[0.2], [-0.1]])
c = 0.25
V = np.array([[0.5, 1]])
W = np.array([[0.8, -0.1], [-0.12, 0.8]])
U = np.array([[2, -1], [1, 1]])

h = np.array([[0.], [0.]])
for t in range(1, 11):
    x = np.array([[np.sin(0.2 * np.pi * t)], [np.cos(0.5 * np.pi * t)]])
    a = b + np.matmul(W, h) + np.matmul(U, x)
    h = np.tanh(a)
    o = c + np.matmul(V, h)
    y = o
    print('y({}) = {}'.format(t, y.tolist()))

x = np.array([[[1], [2]], [[-1], [0]], [[1], [-1]]])
y = np.array([[-1], [1], [2]])
h = [np.array([[0.], [0.]])]
o = []
dy = []
dh = np.array([[0.], [0.]])
for t in range(1, 4):
    xi = x[t - 1]
    yi = y[t - 1][0]
    a = b + np.matmul(W, h[-1]) + np.matmul(U, xi)
    h.append(np.tanh(a))
    o.append(c + np.matmul(V, h[-1]))
    
    # print('h({}) = {} y({}) = {}'.format(t, h.tolist(), t, o.tolist()))
    # print('do({}) = {}'.format(t, -2 * dy[-1]))
    # print('dh({}) = {}'.format(t, dh[-1].tolist()))

dc, db, dv, dw, du = 0, 0, 0, 0, 0
for t in range(3, 0, -1):
    dy = y[t-1] - o[t-1]
    dh1 = -2 * V.T * dy
    diag = np.eye(2) * (np.ones((2, 2)) - h[t] ** 2)
    dh2 = np.matmul(np.matmul(W.T, diag), dh)
    dh = dh1 + dh2
    dc = dc + dy
    db = db + np.matmul(diag, dh)
    dv = dv + dy * h[t].T
    dw = dw + np.matmul(np.matmul(diag, dh), h[t-1].T)
    du = du + np.matmul(np.matmul(diag, dh), x[t-1].T)
print(dc.tolist(), db.tolist(), dv.tolist(), dw.tolist(), du.tolist())