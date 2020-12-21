# -*- coding: utf-8 -*-

"""
・300点自力AC。速攻ではなかったけど。
・式変形
　4/N=1/h+1/n+1/w をwについて解いて、
　w=(N*h*n)/(4*h*n-N*n-N*h)にする。
　あとはhとnで3500までループ回せばOK
・3500*3500=約1200万のループがきつくて、pythonTLEでpypy0.3秒くらいでAC
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

for h in range(1, 3501):
    for n in range(1, 3501):
        # 分母0を避ける
        if 4*h*n-N*n-N*h==0: continue
        w=(N*h*n)/(4*h*n-N*n-N*h)
        # wが正整数になっていればOK
        if w>0 and w.is_integer():
            print(h, n, int(w))
            exit()
