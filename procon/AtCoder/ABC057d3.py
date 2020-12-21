# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC！
"""

import sys
from collections import Counter

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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def nCr(n, r):
    """ 事前テーブルなし組み合わせ簡易版 """
    from math import factorial

    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    return factorial(n) // (factorial(r) * factorial(n-r))

N, L, R = MAP()
A = LIST()

A.sort(reverse=1)
C = Counter(A)
mx = cnt = 0
for i in range(L, R+1):
    A2 = A[:i]
    avg = sum(A2) / i
    # 平均がまだ最大なら続ける
    if avg >= mx:
        mx = avg
    else:
        break
    C2 = Counter(A2)
    # 最後の要素が複数あるなら交換可
    a = A2[-1]
    # 全体から必要な数を選ぶ通り数
    cnt += nCr(C[a], C2[a])
print(mx)
print(cnt)
