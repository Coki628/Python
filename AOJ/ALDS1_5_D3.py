# -*- coding: utf-8 -*-

"""
・蟻本演習3-3-4
・平方分割、転倒数数え上げ、座標圧縮
・計算量20万のN*√Nで約9000万。pythonTLE(制約5秒でも)
"""

import sys
from math import sqrt

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

def compress(A):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(set(A))):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N = INT()
A = LIST()

zipped, _ = compress(A)

# 平方分割
D = ceil(sqrt(N))
bucket = [[] for i in range(D)]

ans = 0
for i in range(N-1, -1, -1):
    a = zipped[A[i]]
    # aが入るバケットより手前のバケットの要素数を数える
    cnt = j = 0
    while j < D and j < a//D:
        cnt += len(bucket[j])
        j += 1
    # aが入るバケット内でaより小さい要素を数える
    for b in bucket[j]:
        if b < a:
            cnt += 1
    ans += cnt
    # aをバケットに詰める
    bucket[a//D].append(a)
print(ans)
