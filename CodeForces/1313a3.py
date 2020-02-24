# -*- coding: utf-8 -*-

"""
・終了後、ツイート見て改善したやつ。
・実は多い方から優先して使えばいいので、
　予め降順ソートさえしておけば、全順番試さなくてもOKだった。
"""

import sys
from collections import Counter
from itertools import combinations

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def check(A, comb):
    for a in comb:
        A[a] -= 1
    if min(A) < 0:
        return False
    else:
        return True

for _ in range(INT()):
    
    A = LIST()
    
    A.sort(reverse=True)
    cnt = 0
    for i in range(1, 4):
        for comb in combinations(range(3), i):
            tmp = A[:]
            if check(A, comb):
                cnt += 1
            else:
                A = tmp
    print(cnt)
