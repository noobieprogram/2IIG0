# Making the imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

df = pd.read_excel('HW3train.xlsx')
x0 = df['X_0'].tolist()
x1 = df['X_1'].tolist()
y = df['y'].tolist()

w1 = [[(random.random()/10)-0.05 for i in range(2)] for j in range(10)]
w2 = [[(random.random()/10)-0.05 for i in range(10)] for j in range(10)]
w3 = [(random.random()/10)-0.05 for i in range(10)]
a=0.5
prevl=2
l=1
size=len(x0)
bs=41
st=0

while st<size: #per batch
    while prevl!=l: #loss changes
        prevl=l
        length = range(st,st+bs)
        w=[]
        for i in length: #forward
            input=[x0[i],x1[i]]
            inner1=[]
            for j in range(10):
                inner1.append(np.maximum(0,np.dot(input,w1[j])))
            inner2=[]
            for j in range(10):
                inner2.append(np.maximum(0,np.dot(inner1,w2[j])))
            output=1/(1+np.exp(-np.dot(inner2,w3)))
            w.append(output)
        l=0
        for i in length: #loss
            l=l+(y[i]-w[i])*(y[i]-w[i])
        l=l/(2*bs)
        print(l)
        
        d1 = [[0]*2]*10
        d2 = [[0]*10]*10
        d3 = [0]*10
        d4=0
        for i in length: #backwards
            d_4=(-2*y[i]+2*w[i])/(2*bs)
            d4=d4+d_4
            d_3=[0]*10
            for j in range(10):
                d_3[j]=(d_4*1/(1+np.exp(-inner2[j]*w3[j]))*(1-1/(1+np.exp(-inner2[j]*w3[j])))*(-inner2[j]))
                d3[j]=d3[j]+d_3[j]
            d_2 = [[0]*10]*10
            for j in range(10):
                for q in range(10):
                    if w2[j][q] > 0:
                        d_2[j][q]=d_3[j]
                    else:
                        d_2[j][q]=0
                    d2[j][q]=d2[j][q]+d_2[j][q]
            d_1 = [[0]*2]*10
            for j in range(10):
                for q in range(2):
                    for p in range(10):
                        if w1[j][q] > 0:
                            d_1[j][q]+=d_2[p][j]
                        else:
                            d_1[j][q]+=0
                    d1[j][q]=d1[j][q]+d_1[j][q]
        
        for i in range(10): #update
            for j in range(2):
                w1[i][j]=w1[i][j]-a*d1[i][j]
        for i in range(10):
            for j in range(10):
                w2[i][j]=w2[i][j]-a*d2[i][j]
        for i in range(10):
            w3[i]=w3[i]-a*d3[i]
    st=st+bs