# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-3、自力AC
"""

import sys

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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, K = MAP()
A = LIST()

cnt = 0
for i in range(1, N):
    # 減らす必要のある数
    tmp = (A[i-1]+A[i]) - K
    if tmp > 0:
        cnt += tmp
        # 右から引けるだけ引く
        A[i] -= min(A[i], tmp)
print(cnt)
