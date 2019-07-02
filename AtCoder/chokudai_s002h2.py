# -*- coding: utf-8 -*-

"""
参考：https://scrapbox.io/snuke/Chokudai_SpeedRun_002
・公式解通り
・式変形
・割る数Xで連立方程式を立てて、2式を引いて文字を減らすと、Xが何の約数なら良いかが分かる。
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
for i in range(N):
    a,b=MAP()
    if a==b:
        print(-1)
    else:
        print(abs(a-b))
