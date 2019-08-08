# -*- coding: utf-8 -*-

"""
・渾身の「各ビット独立で考える」はWA
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

K,M=MAP()
A=LIST()
C=LIST()

# ビット毎に分けて持っておく
A2=[[0]*K for i in range(32)]
C2=[[0]*K for i in range(32)]
for i in range(32):
    for j in range(K):
        if A[j]>>i&1:
            A2[i][j]=1
for i in range(32):
    for j in range(K):
        if C[j]>>i&1:
            C2[i][j]=1

# 最初のK回まで
dp=[[0]*K for i in range(32)]
for i in range(32):
    for j in range(K):
        dp[i][j]=A2[i][j]

# K回より先を各ビット毎に周期になるまで調べる
cycles=[]
for i in range(32):
    cycle=0
    prevs={}
    k=K
    while True:
        # 周期になったか確認
        prev=['']*K
        for j in range(K):
            prev[j]=str(dp[i][k-j-1])
        prev=''.join(prev)
        if prev in prevs:
            idx=prevs[prev]
            # 周期とその開始位置を保持する
            cycles.append((idx, dp[i][idx:-K]))
            break
        # 周期になるまでは普通に進める
        prevs[prev]=k-K
        cycle+=1
        tmp=0
        for j in range(K):
            tmp^=(dp[i][k-j-1]&C2[i][j])
        dp[i].append(tmp)
        k+=1

bit=[0]*32
for i in range(32):
    idx,cycle=cycles[i]
    # Mが小さければ普通に調べた所から
    if M<idx:
        bit[i]=dp[i][M-1]
    else:
        # 大きければ周期から取る
        mod=len(cycle)
        bit[i]=cycle[(M-1-idx)%mod]
ans=0
for i in range(32):
    if bit[i]:
        ans+=1<<i
print(ans)
