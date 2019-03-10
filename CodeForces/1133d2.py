# -*- coding: utf-8 -*-

"""
参考：http://codeforces.com/blog/entry/65807
・本番中はAもBも0の場合を考慮できなかった。
・小数にすると誤差が出るので分数の形のままキーにしてカウントする。
"""

import sys
from collections import Counter
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd 

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N=INT()
A=LIST()
B=LIST()

C=Counter()
cnt0=0
for i in range(N):
    # aもbも0の場合はどのケースにも追加できる
    if A[i]==0 and B[i]==0:
        cnt0+=1
    # Aだけ0だと何もできない
    if A[i]==0:
        continue
    # いわゆる約分はGCD使うと1回で綺麗にできる
    _gcd=gcd(B[i], A[i])
    if B[i]/A[i]>=0:
        is_plus=True
    elif B[i]/A[i]<0:
        is_plus=False
    # キーを(正負、分子、分母)にしてカウントしていく
    C[(is_plus, abs(B[i]//_gcd), abs(A[i]//_gcd))]+=1

mx=0
for k, v in C.items():
    mx=max(mx, v)
print(mx+cnt0)
