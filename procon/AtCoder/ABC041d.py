# -*- coding: utf-8 -*-

"""
参考：http://abc041.contest.atcoder.jp/data/abc/041/editorial.pdf
　　　http://sei0o.hateblo.jp/entry/2017/07/30/113552
・トポロジカルソートの数え上げ
・bitDP
・計算量はこの実装だと多分2^N*N^2で1700万くらい。pypy0.4秒、python2.5秒(3秒OKの問題だったから通った)
・頂点じゃなくて辺ループでやればN^2がMにできるから後でやってみよう。
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

N,M=MAP()
# 有向辺(確定している並び)を隣接行列で持つ
XY=list2d(N, N, False)
for i in range(M):
    x,y=MAP()
    XY[x-1][y-1]=True

# dp[i] := 頂点集合iをトポロジカルソートする方法の通り数
dp=[0]*(1<<N)
# 空集合に1
dp[0]=1

for i in range(1<<N):
    # 集合iに頂点jを追加できるか
    for j in range(N):
        # jは既にiに含まれている
        if (i>>j)&1: continue
        # jとkの関係について確認していく
        for k in range(N):
            # 同じ辺は関係ない
            if j==k: continue
            # kが集合iに含まれていて、jからkに向かう辺があったらNG
            if (i>>k)&1 and XY[j][k]:
                break
        else:
            # 問題なく最後まで行けばOKなので遷移させる
            dp[i+(1<<j)]+=dp[i]
# 全部含めた集合が持っているのが答え
print(dp[-1])
