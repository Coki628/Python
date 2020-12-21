# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc015
・3次元DP、minmaxDP
・計算量O(N*K*W)=50*50*1万=2500万
・この手のやつは逆から回せば前回のやつも必要ないんだった。
・これでpypyで0.6秒。
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

W=INT()
N,K=MAP()
AB=[]
for i in range(N):
    a,b=MAP()
    AB.append((a, b))

# dp[j][k] := i枚目まで考えた時、j枚使って幅kで得られる重要度の最大値
dp=list2d(K+1, W+1, -INF)
dp[0][0]=0
for i in range(N):
    a,b=AB[i]
    # 逆順で回せば前回のも必要ない
    for j in range(K, -1, -1):
        for k in range(W, -1, -1):
            if dp[j][k]!=-INF:
                # K枚以内、幅W以内の条件を満たすならi枚目を使う遷移をさせる
                if j+1<=K and k+a<=W:
                    dp[j+1][k+a]=max(dp[j+1][k+a], dp[j][k]+b)

# N枚目まで見終わった中で最大のものを出力
ans=-INF
for i in range(K+1):
    for j in range(W+1):
        ans=max(ans, dp[i][j])
print(ans)
