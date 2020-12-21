# -*- coding: utf-8 -*-

"""
・1次元数え上げDP
・BITで遷移を高速化
・ここまで来るのにすげー時間かかった…。
・遷移の範囲を探すのを、ちょっと尺取っぽく後ろのidxが追いかけてくる感じにした。
・これでAC。でも絶対もっとやりやすくできたと思う。
・計算量は10万のNlogNで、python1.0秒pypy0.3秒。
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

# サプリiが遡れる場所を格納する
prevs=[0]*(N+1)
prev=0
visited=set()
for i in range(1, N+1):
    # 既に摂取したサプリが来たら後ろを動かす
    if A[i] in visited:
        prev+=1
        while A[prev]!=A[i]:
            visited.remove(A[prev])
            prev+=1
        visited.remove(A[prev])
    prevs[i]=prev
    visited.add(A[i])

# dp[i] := サプリiまで摂取した状態の通り数(BITで構築)
dp=BIT(N+1)
dp.add(0, 1)
for i in range(1, N+1):
    # idxから現在位置までの区間和を取得(まとめて遷移)
    sm=dp.get(prevs[i], i)%MOD
    # dpの現在位置を更新
    dp.add(i, sm)
print(dp.get(N))
