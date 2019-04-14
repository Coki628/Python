# -*- coding: utf-8 -*-

"""
・とりあえず部分点の自力400点
・メモ化再帰
"""

import sys
from bisect import bisect_left, bisect_right

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

N,L=MAP()
# 小数を考慮するため、2倍して考えておく
L*=2
A=[INT()*2 for i in range(N)]
Aset=set(A)
dist=[False]*8000

def rec(cur):
    # 最後のハードルより先まで行けたら成功
    if cur>A[-1]:
        YES()
        exit()
    # 一度見たところはやらない
    if dist[cur]:
        return
    dist[cur]=True
    # 一歩進めるなら進む
    if cur+1 not in Aset:
        rec(cur+1)
    # ジャンプ出来るならする
    if cur+L not in Aset:
        a=bisect_left(A, cur+L)
        b=bisect_right(A, cur+L+L)
        # 跳べない区間にハードルが無いようならいける
        if a==b:
            rec(cur+L+L)

rec(0)
NO()
