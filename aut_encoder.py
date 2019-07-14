import numpy as np
import math
import matplotlib.pyplot as plt


data = np.genfromtxt(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignement 3\MNISTnumImages5000.txt')
dataResult=np.genfromtxt(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignement 3\MNISTnumLabels5000.txt')
data_test = data[4000:,:]
w1 = np.random.rand(100, 784)
w2 = np.random.rand(784, 100)
w1 = w1 * 0.01
w2 = w2 * 0.04
w3 = np.zeros_like(w1)
w4 = np.zeros_like(w2)
eta1 = 0.04
eta2 = 0.04

for m in range(5):
    for n in range(2):
        for i in range(4000):
            s1 = np.zeros(100)
            for a in range(100):
                s1[a] = np.dot(w1[a], data[i])
            h1 = [1/(1+math.exp(-x)) for x in s1]
            s2 = np.zeros(784)
            for b in range(784):
                s2[b] = np.dot(w2[b], h1)
            yout = [(1/(1+math.exp(-x))) for x in s2]
            err = data[i] - yout
            cost_func_train = [(e**2) / 2 for e in err]
            
            for c in range(784):
                for d in range(100):
                    w4[c,d] = (w4[c,d] *0.1) + err[c]*yout[c]*(1-yout[c])*h1[d]*eta1
            w2 = w2+w4
            change = np.zeros(100)
            for e in range(100):
                for f in range(784):
                    change[e] += err[f]*yout[f]*(1-yout[f])*w2[f,e]
            for g in range(100):
                for h in range(784):
                    w3[g,h] = (w3[g,h]*0.1) + eta2 * change[g] * h1[g] * (1-h1[g]) * data[i,h] 
            w1 = w1+w3
    print("Done")


fig = plt.figure(figsize = (7,7))
rows = 10
columns = 10
for x in range(1, 101):
    S1 = np.zeros(100)
    for y in range(100):
        S1[y] = np.dot(w1[y], data_test[x])
    H = [(1/(1+math.exp(-a))) for a in S1]
    S2 = np.zeros(784)
    for z in range(784):
        S2[z] = np.dot(w2[z], H)
    ycap = [1/(1+math.exp(-a)) for a in S2] 
    img = np.reshape(ycap, (28, 28), order = 'F')
    fig.add_subplot(rows, columns, x)   
    plt.imshow(img, cmap = 'gray')
plt.show()                   