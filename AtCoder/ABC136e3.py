# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/kyopro_friends/status/1158010219407892480
・候補を総和の約数に絞るのは悪くなかったみたいだ。
・その先、K回以内の操作でいけるかの判定は自力で探せなかったんだが。
・これはKが大きいのに1つずつ数えてるからTLE
　→各要素について操作をまとめてやるようにしてAC、pythonで0.7秒。
・計算量は約数の数*判定する分=sqrt(N*max(A))*NlogN=sqrt(5億)*500*9=22000*4500=約1億？
　多いな、まあN*max(A)とか実際はもっと全然少ないだろうしいいのか。
"""

import sys
from math import sqrt
from collections import deque

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

N,K=MAP()
A=LIST()

def divisor_set(N: int) -> set:
    """ 約数の列挙・個数 """
    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て
    for i in range(2, int(sqrt(N))+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

# 操作は+1-1で総和が変わらないので、候補はAの総和の約数に限られる
sm=sum(A)
divs=sorted(divisor_set(sm), reverse=True)

for div in divs:
    B=[]
    for i in range(N):
        if A[i]%div!=0:
            B.append(A[i]%div)
    # mod取ったものをソート、これを前後からdivの約数に合わせていく
    B.sort()
    B=deque(B)
    cnt=0
    while len(B):
        # 先頭を引いていってdivの約数になるまでの距離
        dista=B[0]%div
        # 末尾を足していってdivの約数になるまでの距離
        distb=div-B[-1]%div
        # どちらが近いかに応じてリストBを操作する
        if dista>distb:
            B[0]-=distb
            B.pop()
            cnt+=distb
        elif dista<distb:
            B[-1]+=dista
            B.popleft()
            cnt+=dista
        elif dista==distb:
            B.pop()
            B.popleft()
            cnt+=dista
    # 操作回数がK回以内ならOK
    if cnt<=K:
        print(div)
        exit()
