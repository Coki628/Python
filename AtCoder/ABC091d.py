# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc092/editorial.pdf
　　　https://www.youtube.com/watch?v=DqqPuIZvJTk
　　　http://hyoga.hatenablog.com/entry/2018/03/21/231229
　　　https://www.hamayanhamayan.com/entry/2018/03/18/090701
・XORは繰り上がりがないのでビット毎に考えられる。
・ビットが立つ範囲は決まっていて循環しているので、それに合わせて効率よくmodを取る。
・ループを回すのはaNでも、二分探索でbNの値を効率よく調べられる。
・pythonもpypyもTLE。O(29*N*2*logN*4)=2億3200万くらい…無理だよなそれは。
・C++でほぼ同じコード書いて出したら2秒ちょっとでAC。すげぇな。
"""

import sys, re
from collections import deque, defaultdict, Counter
from math import sqrt, hypot, factorial, pi, sin, cos, radians
from heapq import heappop, heappush, heapify, heappushpop
from bisect import bisect_left, bisect_right
from itertools import permutations, combinations, product
from operator import itemgetter, mul
from copy import deepcopy
from functools import reduce, partial
from fractions import Fraction
from string import ascii_lowercase, ascii_uppercase, digits

def input(): return sys.stdin.readline().strip()
def ceil(a, b=1): return int(-(-a // b))
def round(x): return int((x*2+1) // 2)
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
aN = LIST()
bN = LIST()

ans = 0
# 各ビットのループ
for i in range(29):
    T = 2 ** i
    # 今回のビット用に2Tでmodを取ったbNを作る
    bN2 = [0] * N
    for j in range(N):
        bN2[j] = bN[j] % (2*T)
    # 二分探索で範囲内の数を数えるのでソートしておく
    bN2.sort()
    cnt = 0
    # aNのループ
    for j in range(N):
        # aNも同じ位置でmodを取る
        a = aN[j] % (2*T)
        # [T-a,2T-a)と[3T-a,4T-a)の範囲内にあるbの数を数える
        cnt += bisect_left(bN2, 2*T-a) - bisect_left(bN2, T-a)
        cnt += bisect_left(bN2, 4*T-a) - bisect_left(bN2, 3*T-a)
    # XORなので、ビットの立っている場所が奇数個なら1
    if cnt % 2 == 1:
        ans += T
print(ans)
