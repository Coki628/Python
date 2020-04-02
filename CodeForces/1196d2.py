# -*- coding: utf-8 -*-

"""
・自力AC！
・累積和応用
・文字毎、mod 3毎で累積和を取る。
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

def check(l, r):
    res = 0
    # 開始位置をmod 3のどこにするか3通り
    for i in range(3):
        cnt = 0
        # R,G,Bそれぞれの区間和を集める
        for a in range(3):
            cnt += acc[a][(i+a)%3][r] - acc[a][(i+a)%3][l]
        res = max(res, cnt)
    return res

for _ in range(INT()):
    N, K = MAP()
    S = input()

    A = [0] * N
    for i, s in enumerate(S):
        if s == 'G':
            A[i] = 1
        elif s == 'B':
            A[i] = 2
    
    acc = list3d(3, 3, N+1, 0)
    for i, a in enumerate(A):
        acc[a][i%3][i+1] = 1
    for i in range(3):
        for j in range(3):
            acc[i][j] = list(accumulate(acc[i][j]))
    
    ans = 0
    for i in range(N-K+1):
        j = i + K
        ans = max(ans, check(i, j))
    print(K - ans)
