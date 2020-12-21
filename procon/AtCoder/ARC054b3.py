# -*- coding: utf-8 -*-

"""
参考：https://yorikuwa.com/m3308/
　　　https://mathwords.net/sisuhoteisiki#i-3
　　　http://www.geisya.or.jp/~mwm48961/kou2/log2.html
・みんなやってた微分で解く解法
・辿り着くまで色々大変だった。途中式とか色々自分ノート参照。
・指数関数の微分どうやるか知らなかった。
・指数の方程式うまく解けなかった。
・指数にxがある時はうまく変形してから両辺でlog取る。
・他の人のやつでlogが3重くらいになって見えるのは、
　底2のlog使わないで底の変換公式使ってるから。
"""

import sys
from math import log, log2

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

P = float(input())

def calc(x):
    return x + 0.5**(x/1.5) * P

# 微分して、f=0になるxを求める
# x = -3/2 * log2(3/(2*P)*1/log(2))
x = 3/2 * log2(2/3*P*log(2))

print(calc(max(0, x)))
