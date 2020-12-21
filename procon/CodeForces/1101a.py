# -*- coding: utf-8 -*-

Q = int(input())
for i in range(Q):
    l, r, d = map(int, input().split())
    if d < l or d > r:
        print(d)
    else:
        print(r+d-r%d)
