# -*- coding: utf-8 -*-

"""
・まず、3で割り切れる数は桁和が3。mod 3で考えて支障ないので全部mod 3にする。
・桁を1つずつ見ていって、桁和が3にできるところで貪欲に作る。
・手前と組んで3にできるかどうかは、前に作った所以降で1と2が出現済かを持っておけば分かる。
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

A = list(map(int, input()))
N = len(A)

A = [a%3 for a in A]
ans = 0
S = set()
for a in A:
    if a == 0:
        ans += 1
        S = set()
    else:
        if 3 - a in S:
            ans += 1
            S = set()
        else:
            if 1 in S:
                S.add((1+a)%3)
            if 2 in S:
                S.add((2+a)%3)
            S.add(a)
print(ans)
