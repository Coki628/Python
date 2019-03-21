# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/code-thanks-festival-2017-open/editorial.pdf
・XOR系
・数え上げDP
・公式解1：ソートする。
・配列を1次元にする。
・取りうる値が限定的なのでdictにする。
・dictにするならループはkeyのある所だけでいい。
・これもACしたけど意外と速度変わらなかった。
・てかこうなるともうソートいらないし公式解じゃないな。
"""

import sys
from copy import copy
from collections import Counter

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

N,K=MAP()
A=[INT() for i in range(N)]

# dp[i][j] := Aをi番目まで使って、XORがjとなる整数の選び方の個数
dp=Counter()
dp[0]=1

for i in range(N):
    # 配る前の現在の値を保持する
    tmp=copy(dp)
    for k, v in tmp.items():
        dp[k^A[i]]=(dp[k^A[i]]+v)%MOD
print(dp[K])
