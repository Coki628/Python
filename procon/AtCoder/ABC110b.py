# -*- coding: utf-8 -*-

N,M,X,Y = map(int, input().split())
xN = list(map(int, input().split()))
yM = list(map(int, input().split()))

for z in range(-100, 101):
    if X < z <= Y and max(xN) < z and min(yM) >= z:
        print('No War')
        exit()
print('War')