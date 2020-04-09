# -*- coding: utf-8 -*-

"""
・優先度を正しく考えてシミュレーションする。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, bt, ac = MAP()
A = LIST()

mxac = ac
cnt = 0
for a in A:
    if a == 1:
        # 充電できるならするのが最善
        if bt > 0 and ac < mxac:
            bt -= 1
            ac += 1
        # できないなら充電に使えない方から使う
        elif ac > 0:
            ac -= 1
        elif bt > 0:
            bt -= 1
        else:
            break
    else:
        if ac > 0:
            ac -= 1
        elif bt > 0:
            bt -= 1
        else:
            break
    cnt += 1
print(cnt)
