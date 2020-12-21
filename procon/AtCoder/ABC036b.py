# -*- coding: utf-8 -*-

import numpy as np

N = int(input())

NN = []
for i in range(N):
    NN.append(list(input()))
NN = np.array(NN)
# numpyに便利なやつあった
NN = np.rot90(NN, k=-1)

for i in range(N):
    print(''.join(NN[i]))