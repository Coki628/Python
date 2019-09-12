# -*- coding: utf-8 -*-

"""
・自力AC！
・3頂点が作る三角形の面積
・Nが小さいので全組み合わせ試す。
・最初WAで、原因は小数の誤差だったと思われる。
・直接/2しないで%2で偶奇判定する形にしたらAC。これだと整数のままだもんね。
"""

import sys
from itertools import combinations

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

N = INT()
XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))

# 頂点s,t,uが作る三角形の面積
def calc(s, t, u):
    x1, y1 = s
    x2, y2 = t
    x3, y3 = u
    # 面積*2
    S2 = abs((x1-x3)*(y2-y3) - (x2-x3)*(y1-y3))
    # 0ではない整数
    return S2 % 2 == 0 and S2 != 0

cnt = 0
# 3点の全組み合わせ
for comb in combinations(XY, 3):
    if calc(comb[0], comb[1], comb[2]):
        cnt += 1
print(cnt)
