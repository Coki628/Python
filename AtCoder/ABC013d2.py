# -*- coding: utf-8 -*-

"""
・D自力AC
・ダブリング
・1回目の結果だけ取ったら、あとはダブリングでD回分進められる。
・自力で通したけど、ダブリングやったばっかだったし、
　ここでダブリング使うの、ABC014dの記事で軽くネタバレしてたからな。。
・計算量はNlogD+M=10万*30くらい(log10億)+20万で320万、python1.8秒pypy0.4秒AC。
・外ループを軽くした方が速くなる話があったので実験。
　がしかし、ほぼ変わらず。。これなら分かりやすいループにした方がいいね。
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

N,M,D=MAP()
A=LIST()

B=list(range(N))
# 1回目のあみだくじ結果を出す
for i in range(M):
    idx=A[i]
    B[idx-1],B[idx]=B[idx],B[idx-1]

# nxt[k][i] := 縦線iからスタートして、2^k回のあみだくじの後にいる位置
nxt=list2d(31, N, -1)
# 初期化
for i in range(N):
    # Bi番目の縦線スタートは、1(2^0)回終わってiにいる
    nxt[0][B[i]]=i
# ダブリングのテーブル構築
for k in range(1, 31):
    for i in range(N):
        nxt[k][i]=nxt[k-1][nxt[k-1][i]]

ans=list(range(N))
for k in range(31):
    for i in range(N):
        # ビットが立っている所に合わせて、移動させていく
        if D>>k&1:
            ans[i]=nxt[k][ans[i]]
[print(a+1) for a in ans]
