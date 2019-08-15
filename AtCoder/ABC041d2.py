# -*- coding: utf-8 -*-

"""
参考：http://abc041.contest.atcoder.jp/data/abc/041/editorial.pdf
　　　http://sei0o.hateblo.jp/entry/2017/07/30/113552
　　　https://atcoder.jp/contests/abc041/submissions/5829678
・トポロジカルソートの数え上げ
・bitDP
・辺でやるのなんかむずかったから、頂点ループのまま1つ減らした。
・隣接行列だったところを、実質2次元目的な部分にbitで値を持たせると
　内ループ回さないで全頂点とチェックができる。
・これでpythonでも0.4秒
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
# XY[i] := 頂点iから出る有向辺を持つ頂点の集合
XY=[0]*N
for i in range(M):
    x,y=MAP()
    XY[x-1]+=1<<(y-1)

# dp[i] := 頂点集合iをトポロジカルソートする方法の通り数
dp=[0]*(1<<N)
# 空集合に1
dp[0]=1

for i in range(1<<N):
    # 集合iに頂点jを追加できるか
    for j in range(N):
        # jは既にiに含まれている
        if (i>>j)&1: continue
        # 2つの集合で&を取れば、重複の有無を確認できる
        if (i&XY[j])==0:
            # 集合iの中に、jから出る頂点がなければOKなので遷移させる
            dp[i+(1<<j)]+=dp[i]
# 全部含めた集合が持っているのが答え
print(dp[-1])
