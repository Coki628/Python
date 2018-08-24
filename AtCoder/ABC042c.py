# -*- coding: utf-8 -*-

N,K = map(int, input().split())
Dk = list(map(str, input().split()))

for i in range(N, 100000):
    i = str(i)
    for j in range(K):
        if i.find(Dk[j]) != -1:
            break
        if j == K-1:
            print(i)
            exit()