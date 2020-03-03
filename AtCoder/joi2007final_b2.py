# -*- coding: utf-8 -*-

"""
・二分探索、ローリングハッシュ
・かえって遅くなった。。。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """

    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

class RollingHash:

    MOD = 2 ** 64
    b = 10 ** 8 + 7

    def __init__(self, S):
        # 各文字を数値に変換しておく
        S = [ord(s)-97 for s in S]

        self.len = len(S)
        self.pow = [1] * (self.len+1)
        for i in range(self.len):
            self.pow[i+1] = self.pow[i] * self.b
            self.pow[i+1] %= self.MOD
        # ハッシュの生成
        self.hash = [0] * (self.len+1)
        for i in range(self.len):
            self.hash[i+1] = self.hash[i] * self.b + S[i]
            self.hash[i+1] %= self.MOD
    
    # 区間[i,j)のハッシュ値を取得
    def get(self, l, r):
        return (self.hash[r] - self.hash[l] * self.pow[r-l]) % self.MOD

S = input()
N = len(S)
T = input()
M = len(T)

# ローリングハッシュ
rh1 = RollingHash(S)
rh2 = RollingHash(T)

# Sのm文字からなる連続部分列がTに含まれるか
def check(m):
    for i in range(N-m+1):
        s = rh1.get(i, i+m)
        for j in range(M-m+1):
            t = rh2.get(j, j+m)
            if s == t:
                return True
    return False

# 二分探索
res = bisearch_max(0, N+1, check)
print(res)
