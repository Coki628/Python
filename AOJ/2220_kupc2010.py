# -*- coding: utf-8 -*-

"""
参考：https://mathtrain.jp/sanjihanbetsu
　　　https://mathtrain.jp/descartes
・蟻本演習3-1-2
・3次方程式の判別式、デカルトの符号法則
・自分で色々調べながらやって、WAでテストケース調べたりはしたけど最終的にAC。
・想定解ではないと思う。
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
INF = float('inf')
MOD = 10 ** 9 + 7

# デカルトの符号法則
def check(a, b, c, d):
    A = [a, b, c, d]
    B = [0] * 4
    for i, x in enumerate(A):
        if x > 0:
            B[i] = 1
        elif x < 0:
            B[i] = -1

    cur = B[0]
    k = 0
    for i in range(1, 4):
        if B[i] != 0 and cur != B[i]:
            cur = B[i]
            k += 1
    poscnt = k

    B[0] *= -1
    B[2] *= -1
    cur = B[0]
    k = 0
    for i in range(1, 4):
        if B[i] != 0 and cur != B[i]:
            cur = B[i]
            k += 1
    negcnt = k

    return (poscnt, negcnt)

# 3次方程式の判別式
def check2(a, b, c, d):
    D = -4*a*c**3 - 27*a**2*d**2 + b**2*c**2 + 18*a*b*c*d - 4*b**3*d
    return D

for _ in range(INT()):
    a, b, c, d = MAP()

    D = check2(a, b, c, d)

    # 符号の変化回数から解の個数(の候補)を取得
    poscnt, negcnt = check(a, b, c, d)

    # 判別式から解が1つなら、それに合わせて個数を決める
    if D < 0:
        if poscnt >= 2:
            poscnt -= 2
        if negcnt >= 2:
            negcnt -= 2

    print(poscnt, negcnt)
