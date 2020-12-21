# -*- coding: utf-8 -*-

N,M,X,Y = map(int, input().split())
xN = list(map(int, input().split()))
yM = list(map(int, input().split()))
# 首都も同一視できる
xN.append(X)
yM.append(Y)

# これだけでも干渉するか分かる
if max(xN) < min(yM):
    print('No War')
else:
    print('War')