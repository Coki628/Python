# -*- coding: utf-8 -*-

"""
・1次元数え上げDP
・BITで遷移を高速化
・一番最初にWAになったやつを、idxとlastでmax取るだけ追加したらACしたー。
・色々遠回りしてたね。。
"""

import sys

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

class BIT:

    def __init__(self, n):
        # 0-indexed
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [0] * nv

    def sum(self, i):
        """ [0, i]を合計する """
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    def add(self, i, x):
        """ 値の追加：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

    def get(self, l, r=None):
        """ 区間和の取得 [l, r) """
        # 引数が1つなら一点の値を取得
        if r is None: r = l + 1
        res = 0
        if r: res += self.sum(r-1)
        if l: res -= self.sum(l-1)
        return res

N,M=MAP()
A=[0]+[INT() for i in range(N)]

# dp[i] := サプリiまで摂取した状態の通り数(BITで構築)
dp=BIT(N+1)
dp.add(0, 1)
# サプリiが最後に出現した場所を保持
last=[0]*(M+1)
# 貰う遷移をどこまで遡ってやるかのidx
idx=0
for i in range(1, N+1):
    # サプリiが最後に出現した場所を取得(前回idxとmax取る)
    idx=max(idx, last[A[i]])
    # 上記場所を現在位置に更新
    last[A[i]]=i
    # idxから現在位置までの区間和を取得(まとめて遷移)
    sm=dp.get(idx, i)%MOD
    # dpの現在位置を更新
    dp.add(i, sm)
print(dp.get(N))
