# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day1/editorial-G.pdf
　　　https://atcoder.jp/contests/iroha2019-day1/submissions/5713292
・いわゆる普通のminmax系DP(2次元+内ループ)
・これは貰うDP実装
・内ループをセグメント木(RMaxQ)にした。
・線形がlogになったはずだけど、かえって遅くなった…。
・セグ木をより汎用的に使えるように改修
・C++でAC確認。
・速度改善版セグ木検証。pypyTLE => 0.5秒AC！
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

class SegTree:
    """
    以下のクエリを処理する
    1.update:  i番目の値をxに更新する
    2.query: 区間[l, r)の値を得る
    """

    def __init__(self, n, func, init):
        """
        :param n: 要素数
        :param func: 値の操作に使う関数(min, maxなど)
        :param init: 要素の初期値
        """
        self.n = n
        self.func = func
        self.init = init
        # nより大きい2の冪数
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.init] * (n2 << 1)
 
    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = x
        while i > 1:
            self.tree[i >> 1] = x = self.func(x, self.tree[i ^ 1])
            i >>= 1
 
    def query(self, a, b):
        """
        [a, b)の値を得る
        :param a: index(0-indexed)
        :param b: index(0-indexed)
        """
        l = a + self.n2
        r = b + self.n2
        s = self.init
        while l < r:
            if r & 1:
                r -= 1
                s = self.func(s, self.tree[r])
            if l & 1:
                s = self.func(s, self.tree[l])
                l += 1
            l >>= 1
            r >>= 1
        return s

N,M,K=MAP()
A=[0]+LIST()

# dp[i][j] := i回目をj日目に使った状態(2次元目をセグ木で構築)
dp=[None]*(M+1)
for i in range(M+1):
    dp[i]=SegTree(N+1, max, -INF)
dp[0].update(0, 0)

for i in range(1, M+1):
    for j in range(1, N+1):
        # 区間maxをセグ木から取得
        mx=dp[i-1].query(max(j-K, 0), j)+A[j]
        dp[i].update(j, mx)

# M回終わって、あとK日で終わりまで行ければOK
ans=dp[M].query(N+1-K, N+1)
if ans!=-INF:
    print(ans)
else:
    print(-1)
