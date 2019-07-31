# -*- coding: utf-8 -*-

"""
・D自力AC
・3次元DP、minmaxDP
・計算量O(N*K*W)=50*50*1万=2500万
・pypyのMLE対策として1次元目を使い捨てで2次元DPにしてAC、0.9秒。
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

def deepcopy(li): return [x[:] for x in li]

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
    # i枚目を使わない遷移
    prev=deepcopy(dp)
    for j in range(K+1):
        for k in range(W+1):
            if prev[j][k]!=-INF:
                # K枚以内、幅W以内の条件を満たすならi枚目を使う遷移をさせる
                if j+1<=K and k+a<=W:
                    dp[j+1][k+a]=max(dp[j+1][k+a], prev[j][k]+b)

# N枚目まで見終わった中で最大のものを出力
ans=-INF
for i in range(K+1):
    for j in range(W+1):
        ans=max(ans, dp[i][j])
print(ans)
