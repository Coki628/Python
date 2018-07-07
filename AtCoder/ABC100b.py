# -*- coding: utf-8 -*-
 
D, N = map(int, input().split())

if N == 100:
    print(100 ** D * N + 100 ** D)
else:
    print(100 ** D * N)