# -*- coding: utf-8 -*-

"""
・ABC022dでACしたやつ、こっちでもやっておく。
・凸包
・アンドリューのアルゴリズム(Monotone Chain)
"""

import sys
from collections import deque
from math import hypot
from operator import itemgetter

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
A=[]
for i in range(N):
    x,y=MAP()
    A.append((x, y))

# yでソート済を準備しておく
Ay=sorted(A, key=itemgetter(0))
Ay.sort(key=itemgetter(1))

# ベクトルの外積
def cross(a, b):
    x1,y1=a
    x2,y2=b
    return x1*y2-y1*x2

def sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

def calc(a, b, c):
    # aから見たb,cへ向かうベクトルにする
    v1=sub(b, a)
    v2=sub(c, a)
    return cross(v1, v2)

# 2点間の距離
def get_dist(v1, v2):
    return hypot(v1[0]-v2[0], v1[1]-v2[1])

# アンドリューのアルゴリズム(Monotone Chain)
def monotone_chain(li):
    # 左半分
    # 使う座標を保持
    stack=deque()
    stack.append(li[0])
    stack.append(li[1])
    for i in range(2, N):
        # 1つ前->次 と 1つ前->2つ前 のベクトルで外積をチェックして向きを判定する
        while len(stack)>=2 and calc(stack[-1], li[i], stack[-2])<0:
            # 次が時計回り側にある時は1つ前を外す
            stack.pop()
        # 次が反時計回り側にあればOKなので進める
        stack.append(li[i])
    res=list(stack)[:-1]

    # 右半分(やることは同じ)
    stack=deque()
    stack.append(li[-1])
    stack.append(li[-2])
    for i in range(N-3, -1, -1):
        while len(stack)>=2 and calc(stack[-1], li[i], stack[-2])<0:
            stack.pop()
        stack.append(li[i])
    res+=list(stack)[:-1]

    return res

ans=monotone_chain(Ay)
print(len(ans))
[print(x, y) for x, y in ans]
