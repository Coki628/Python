# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc030
・桁DP
・やばい大きさの値でmod取りたい時は、桁毎に前から順に処理できる。
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

N,a=MAP()
a-=1
K=INT()
B=[b-1 for b in LIST()]

li=[]
x=0
visited=[False]*N
# サイクルになるまで再帰する
def rec(cur):
    if visited[cur]:
        global x
        # サイクルの開始位置が分かるように取っておく
        x=cur
        return
    visited[cur]=True
    li.append(cur)
    res=rec(B[cur])
    return

rec(a)

idx=li.index(x)
firstlen=len(li[:idx])
li2=li[idx:]
cyclelen=len(li2)

# Kが小さければそのまま
if K<=firstlen:
    print(li[K]+1)
else:
    # 大きければサイクルの大きさでmod取る
    mod=len(li2)
    K=str(K)
    # dp[i] := 左からi桁だけみた時のKの値
    dp=[0]*(len(K)+1)
    # 大きい桁から順番に見ていく
    for i in range(len(K)):
        # 毎回mod取れば、値が大きくならないで済む(*+しかしないからいつでもmod取れる)
        dp[i+1]=(dp[i]*10+int(K[i]))%mod
    print(li2[(dp[-1]-firstlen)%mod]+1)
