# -*- coding: utf-8 -*-

"""
・久々の500点自力AC！
・結構さくっと解けたし。
・状況整理して数を減らしたビット全探索
"""

import sys
from collections import Counter
from copy import copy

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
D=LIST()

# 自分の位置で0は確定
L=[0]
C=Counter(D)
S=[]
# とりあえず12までしかないので集計
for k, v in C.items():
    # 3人以上いたら、2人は絶対同じ場所になる
    if v>=3:
        print(0)
        exit()
    # 2人いたら、Dと24-Dにそれぞれ置く
    elif v>=2:
        L.append(k)
        L.append(24-k)
    # 1人ならビット全探索用に別保管
    else:
        S.append(k)

mx=0
# これだけ減ればビット全探索できるはず
for i in range(1<<len(S)):
    L2=copy(L)
    for j in range(len(S)):
        # ビットの状態によって、Dか24-Dに置く
        if i>>j&1:
            L2.append(S[j])
        else:
            L2.append(24-S[j])
    # 量少ないからソートもさして影響ないはず
    L2.sort()
    # 循環するから2つ繋げる
    L2=L2+[l+24 for l in L2]
    mn=INF
    # 隣接する場所の距離を取るループ
    for i in range(1, len(L2)):
        # このビットの状態での最小値を決める
        # 24離れたら同じ場所なのでmod 24しておく
        mn=min(mn, abs(L2[i-1]-L2[i])%24)
    # 全部の状態を試した最大値を探す
    mx=max(mx, mn)
print(mx)
