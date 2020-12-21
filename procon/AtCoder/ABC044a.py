# -*- coding: utf-8 -*-

N = int(input())
K = int(input())
X = int(input())
Y = int(input())

if N - K > 0:
    print(K * X + ((N - K) * Y))
else:
    print(N * X)