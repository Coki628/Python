# -*- coding: utf-8 -*-

"""
参考：https://cf16-final.contest.atcoder.jp//data/other/cf16-exhibition-final/editorial.pdf
　　　https://mathtrain.jp/heron
　　　https://mathtrain.jp/naisetsuen
・数学系
・ヘロン、内接円
・考え方は公式解説と自分のノート参照
"""

import sys
from math import hypot, sqrt

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

def heron(a, b, c):
    s=(a+b+c)/2
    return sqrt(s*(s-a)*(s-b)*(s-c))

x1,y1=MAP()
x2,y2=MAP()
x3,y3=MAP()

# 各辺の長さ
a=hypot(x1-x2, y1-y2)
b=hypot(x2-x3, y2-y3)
c=hypot(x3-x1, y3-y1)
# 三角形の面積(ヘロンの公式)
S=heron(a, b, c)
# 内接円の半径
r=2*S/(a+b+c)

# max(a,b,c)*(1-x/r)==2*xをxについて解く
mx=max(a, b, c)
print(mx/(2+mx/r))
