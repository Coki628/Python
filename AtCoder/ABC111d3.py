# -*- coding: utf-8 -*-

"""
参考：https://betrue12.hateblo.jp/entry/2018/09/30/125042
　　　https://www.youtube.com/watch?v=fYS-rAUSD5o
・2冪、座標変換
・アームを2冪で構成するとどこでも行けるようにできる。
・斜めに交差するu座標とv座標を定義すると、向けたいアームの方向の見通しがつくようになる。
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
XY=[[None] for i in range(N)]
x,y=MAP()
XY[0]=(x, y)
od_ev=(abs(x)+abs(y))%2
for i in range(1, N):
    x,y=MAP()
    # まず大前提として偶奇が合うかどうか
    if od_ev!=(abs(y)+abs(x))%2:
        print(-1)
        exit()
    XY[i]=(x, y)

# アームの作成(降順の2冪で構成する)
arms=[]
for i in range(31, -1, -1):
    arms.append(2**i)
# 偶数の場合は1を加えて位置調整に使う
if od_ev==0: arms.append(1)
print(len(arms))
print(*arms)

for i in range(N):
    ans=['']*len(arms)
    x,y=XY[i]
    # 斜めに交差するu座標とv座標を使う
    u=x+y
    v=x-y
    # 大きい方から貪欲に決めて行けばOK
    for j, arm in enumerate(arms):
        # uとvの正負の応じて、動かす方向を決める
        # 最終的にuとvは0に収束する
        if u>=0 and v>=0:
            ans[j]='R'
            u-=arm
            v-=arm
        elif u>=0 and v<0:
            ans[j]='U'
            u-=arm
            v+=arm
        elif u<0 and v>=0:
            ans[j]='D'
            u+=arm
            v-=arm
        elif u<0 and v<0:
            ans[j]='L'
            u+=arm
            v+=arm
    print(''.join(ans))
