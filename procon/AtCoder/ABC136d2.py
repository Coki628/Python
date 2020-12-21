# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc136/editorial.pdf
・別解、ダブリング、最近ちょうどやったテーマだったし練習。
・これなら各マス個別に任意の回数で聞かれてもlogNで答えられて汎用的。
・10万のNlogNで、pythonで0.9秒AC。
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

S=input()
N=len(S)

# nxt[k][i] := 最初iにいた子が2^k回の移動後にいるマス
nxt=list2d(17, N, -1)
# 初期化
for i in range(N):
    if S[i]=='L':
        nxt[0][i]=i-1
    elif S[i]=='R':
        nxt[0][i]=i+1
# ダブリングのテーブル構築
for k in range(1, 17):
    for i in range(N):
        nxt[k][i]=nxt[k-1][nxt[k-1][i]]

ans=[0]*N
# 10万回移動後を求める(今回はこれで10**100と一致する)
K=10**5
# 元々iにいた子
for i in range(N):
    cur=i
    for k in range(17):
        # ビットが立っている所に合わせて、移動させていく
        if K>>k&1:
            cur=nxt[k][cur]
    ans[cur]+=1
print(*ans)
