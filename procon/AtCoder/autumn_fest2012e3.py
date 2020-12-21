# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/tomerun/together-14867665
・実験から、条件を推測できる。
・各ターンにおいて、最大距離が足りていて、最大距離と偶奇が一致していれば、
　実は短い距離は他の制限はなく自由に作れる。
・後はそれに基づいて条件を満たす最初のターンを見つければOK。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
YX = []
for i in range(N):
    x, y = MAP()
    YX.append((y, x))

# 最大距離の取得と偶奇一致の確認
mx = 0
mod2 = (abs(YX[0][0]) + abs(YX[0][1])) % 2
for y, x in YX:
    cur = abs(y) + abs(x)
    if cur % 2 != mod2:
        print(-1)
        exit()
    mx = max(mx, cur)

i = 1
while 1:
    cur = i*(i+1) // 2
    if cur % 2 == mod2 and cur >= mx:
        # 最初に条件を満たしたターンを出力
        print(i)
        exit()
    i += 1
