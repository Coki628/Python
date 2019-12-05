# -*- coding: utf-8 -*-

"""
・結構すんなり自力AC！
・ゲーム系、Nimに帰着、Grundy数
・考察が真っ直ぐ正しい方向に進んで、良い感じだった。
・やはり3次元はほぼブラフ。個別の面(というか線)で6回見るだけ。
・各豆腐の食べ進める6方向それぞれを、Nimの1つの山と考えられる。
・後は複数山Nimと全く同じ手順で、全ての山のGrundy数をXORすればOK。
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
ans = 0
for i in range(N):
    x, y, z = MAP()
    M = INT()
    # 各次元について、ハバネロの左端と右端を確認する
    lx = ly = lz = INF
    rx = ry = rz = -INF
    for j in range(M):
        x2, y2, z2 = MAP()
        lx = min(lx, x2)
        ly = min(ly, y2)
        lz = min(lz, z2)
        rx = max(rx, x2+1)
        ry = max(ry, y2+1)
        rz = max(rz, z2+1)
    # 6方向それぞれから、何回食べ進められるか(Grundy数)
    ans ^= lx
    ans ^= ly
    ans ^= lz
    ans ^= x - rx
    ans ^= y - ry
    ans ^= z - rz
# 全豆腐全方向からのGrundy数XORで、0でなければ先手番の勝ち
if ans != 0:
    print('WIN')
else:
    print('LOSE')
