# -*- coding: utf-8 -*-

N,Q = map(int, input().split())

Ai = [0] * N

for i in range(Q):
    L,R,T = map(int, input().split())
    for j in range(L-1, R):
        Ai[j] = T

for i in range(N):
    print(Ai[i])