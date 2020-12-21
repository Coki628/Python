# -*- coding: utf-8 -*-

"""
貪欲(解説参考)
"""

N = int(input())
NG = [0] * 3
NG[0] = int(input())
NG[1] = int(input())
NG[2] = int(input())

if N in NG:
    print('NO')
    exit()
cur = N
for i in range(100):
    if cur - 3 not in NG:
        cur -= 3
    elif cur - 2 not in NG:
        cur -= 2
    elif cur - 1 not in NG:
        cur -= 1
    else:
        print('NO')
        exit()
if cur <= 0:
    print('YES')
else:
    print('NO')
