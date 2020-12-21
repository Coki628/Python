# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day1/editorial-G.pdf
　　　https://atcoder.jp/contests/iroha2019-day1/submissions/5713292
・いわゆる普通のminmax系DP(2次元+内ループ)
・これは貰うDP実装
・内ループをセグメント木(RMaxQ)にした。
・線形がlogになったはずだけど、かえって遅くなった…。
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

class SegTreeMax:
    """
    以下のクエリを処理する
    1.update:  i番目の値をxに更新する
    2.get_max: 区間[l, r)の最大値を得る
    """
 
    def __init__(self, n):
        """
        :param n: 要素数
        """
        self.n = n
        # nより大きい2の冪数
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [-INF] * (n2 << 1)
 
    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = x
        while i > 1:
            self.tree[i >> 1] = x = max(x, self.tree[i ^ 1])
            i >>= 1
 
    def get_max(self, a, b):
        """
        [a, b)の最大値を得る
        :param a: index(0-indexed)
        :param b: index(0-indexed)
        """
        return self._get_max(a, b, 1, 0, self.n2)
 
    def _get_max(self, a, b, k, l, r):
        """
        [a, b)の最大値を得る内部関数
        :param k: 現在調べている区間のtree内index
        :param l, r: kが表す区間の左右端index [l, r)
        :return: kが表す区間と[a, b)の共通区間内での最小値。共通区間を持たない場合は-INF
        """
        # 範囲外ならINF
        if r <= a or b <= l:
            return -INF
        # [a,b)が完全に[l,r)を包含するならtree[k]の値を採用
        if a <= l and r <= b:
            return self.tree[k]
        # 一部だけ範囲内なら2つに分けて再帰的に調査
        m = (l + r) // 2
        return max(
            self._get_max(a, b, k << 1, l, m),
            self._get_max(a, b, (k << 1) + 1, m, r)
        )

N,M,K=MAP()
A=[0]+LIST()

# dp[i][j] := i回目をj日目に使った状態(2次元目をセグ木で構築)
dp=[None]*(M+1)
for i in range(M+1):
    dp[i]=SegTreeMax(N+1)
dp[0].update(0, 0)

for i in range(1, M+1):
    for j in range(1, N+1):
        # 区間maxをセグ木から取得
        mx=dp[i-1].get_max(max(j-K, 0), j)+A[j]
        dp[i].update(j, mx)

# M回終わって、あとK日で終わりまで行ければOK
ans=dp[M].get_max(N+1-K, N+1)
if ans!=-INF:
    print(ans)
else:
    print(-1)
