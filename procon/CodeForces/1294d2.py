# -*- coding: utf-8 -*-

"""
・TLE後、C++でセグ木にぶたんACしたり頑張ったんだけど、
　よく見たら最小値のindexは最小値の同値が増えた時に都度確認すればlogすらいらなかった。
・これでpypyも自力AC。終了後1時間以上経ってからのことだけどね。。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

Q, X = MAP()
# Xを使って自由に操作できるので、mod Xで同値とみなしていい
A = [a%X for a in LIST(Q)]

# クエリを逆から見ていく
A = A[::-1]
C = Counter(A)

mn = min([C[a] for a in range(X)])
add = 0
for a in range(X):
    if C[a] == mn:
        add = a
        break
base = mn * X

ans = []
for a in A:
    ans.append(base + add)
    C[a] -= 1
    # 最小個数の同値が来たら値の最小を確認
    if C[a] == mn:
        add = min(add, a)
    # 最小個数を更新する時は値も更新
    elif C[a] < mn:
        mn -= 1
        base -= X
        add = a
[print(a) for a in ans[::-1]]
