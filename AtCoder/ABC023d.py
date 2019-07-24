# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc023
・ちょっと式変形すると、ある高さまでに割りたい時、いつまでに割らないといけないかが分かる。
　風船を割った高さ = h+s*割った時間
　→風船を割った時間 = (割った高さ-h)/s
・これで後は高さを決めての二分探索でOK。
・計算量が、中の計算(Nとソート)*二分探索でO(NlogN*log(H+S*N))で10万*17くらい*50くらい
　で、8500万くらいになる。
・pythonで3.6秒(この問題は制限5秒)、pypyで1.1秒。
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

N=INT()
HS=[]
for i in range(N):
    h,s=MAP()
    HS.append((h, s))

A=[0]*N
def calc(m):
    for i in range(N):
        h,s=HS[i]
        # 風船を割った時間 = (風船を割った高さ-h)/s
        A[i]=(m-h)/s
    A.sort()
    for i in range(N):
        # 1つでも経過時間より早く割らないといけない風船があったらNG
        if A[i]<i:
            return False
    return True

# 二分探索(その高さで全ての風船を割れるか)
ok=10**9+10**9*N
ng=0
while ng+1<ok:
    mid=(ok+ng)//2
    if calc(mid):
        # 下を探しに行く
        ok=mid
    else:
        # 上を探しに行く
        ng=mid
print(ok)
