# -*- coding: utf-8 -*-

"""
参考：http://arc052.contest.atcoder.jp/data/arc/052/editorial.pdf
・円錐…自力ではさっぱり解法出ず、やり方見たらすぐできた。
・3次元に惑わされないこと。実際、今回立体であることを意識する必要はほぼない。
・三角形の面積とほぼ同じに考えれば、普通に累積和取って区間和クエリ処理する感じになる。
"""

import sys
from math import pi
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

# 円錐の体積
def get_cone(r, h):
    return pi*r**2*h/3

N, Q = MAP()
MAX = 2*10**4+1
cones = list2d(N, MAX, 0)
for i in range(N):
    x, r, h = MAP()
    prev = get_cone(r, h)
    # 各円錐の高さ1毎の体積
    for h2 in range(h-1, -1, -1):
        # 高さに合わせて円の半径も小さくしていく
        r2 = r * h2 / h
        cur = get_cone(r2, h2)
        cones[i][x+h-h2] = prev - cur
        prev = cur

sums = [0] * MAX
# 行列入れ替え
cones = list(zip(*cones))
# 各高さ毎に全円錐を合計する
for i in range(MAX):
    sums[i] = sum(cones[i])
# 区間和取得のために累積和しておく
sums = list(accumulate(sums))

for i in range(Q):
    a, b = MAP()
    print(sums[b]-sums[a])
