# -*- coding: utf-8 -*-

H1,W1 = map(int, input().split())
H2,W2 = map(int, input().split())

if H1 == H2 or H1 == W2 or H2 == W1 or W1 == W2:
    print('YES')
else:
    print('NO')