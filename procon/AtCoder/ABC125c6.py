# -*- coding: utf-8 -*-

"""
・蟻本演習3-3-4
・平方分割
・計算量はN<=10万*√Nで約3200万に、今回GCDのlogもかかってるからつらいみたい。
・pypyTLEで、C++でAC確認0.5秒。
"""

import sys
from math import sqrt
from functools import reduce
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd

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

def gcd_list(li): return reduce(gcd, li, 0)

N = INT()
A = LIST()

# 平方分割
D = ceil(sqrt(N))
bucket = [[] for i in range(D)]
gcds = [0] * D
for i in range(N):
    bucket[i//D].append(A[i])
# 各バケットで全体のGCDを予め取っておく
for i in range(D):
    gcds[i] = gcd_list(bucket[i])

ans = 0
for i in range(D):
    for j in range(len(bucket[i])):
        tmp = 0
        # バケット内でa以外全ての要素とGCD
        for k, a in enumerate(bucket[i]):
            if j == k:
                continue
            tmp = gcd(tmp, a)
        # aを含むバケット以外の全てのバケットとGCD
        for k in range(D):
            if i == k:
                continue
            tmp = gcd(tmp, gcds[k])
        ans = max(ans, tmp)
print(ans)
