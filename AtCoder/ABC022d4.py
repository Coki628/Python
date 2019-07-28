# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc022
　　　AOJ本P.401~
　　　https://matsu7874.hatenablog.com/entry/2018/12/17/025713
・変化前後の各点の対応関係を具体的に求める必要はない。
・比だけが知りたいので、移動や回転をしても変わらない何かを使う。
・凸包の外周の長さ
・アンドリューのアルゴリズム(Monotone Chain)
・これもO(NlogN)、pypyで1.2秒のAC
・スタック2つだったのを1つにまとめた。(速度は特に変化なし)
"""

import sys
from collections import deque
from math import hypot

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
B=[]
for i in range(N):
    x,y=MAP()
    B.append((x, y))

# xでソート済を準備しておく
Ax=sorted(A)
Bx=sorted(B)

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
    # 上半分
    # 使う座標と距離を保持
    stack=deque()
    stack.append((li[0], 0))
    stack.append((li[1], get_dist(li[0], li[1])))
    for i in range(2, N):
        # 1つ前->次 と 1つ前->2つ前 のベクトルで外積をチェックして向きを判定する
        while len(stack)>=2 and calc(stack[-1][0], li[i], stack[-2][0])>0:
            # 次が反時計回り側にある時は1つ前を外す
            stack.pop()
        # 次が時計回り側にあればOKなので進める
        stack.append((li[i], get_dist(stack[-1][0], li[i])))
    # 上半分の長さを合計
    res=sum([dist for v, dist in stack])

    # 下半分(やることは同じ)
    stack=deque()
    stack.append((li[-1], 0))
    stack.append((li[-2], get_dist(li[-1], li[-2])))
    for i in range(N-3, -1, -1):
        while len(stack)>=2 and calc(stack[-1][0], li[i], stack[-2][0])>0:
            stack.pop()
        stack.append((li[i], get_dist(stack[-1][0], li[i])))
    res+=sum([dist for v, dist in stack])

    return res

a=monotone_chain(Ax)
b=monotone_chain(Bx)
print(b/a)
