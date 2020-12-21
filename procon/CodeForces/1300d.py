# -*- coding: utf-8 -*-

"""
・終了後10分くらいで書いたやつで自力AC。。こどふぉこういうの多いな。
・結局YESなのって全方向に対称な図形だけっぽくない？からの各座標確認でOKだった。
・これでいいならEで悩んでないで先にちゃんとこっち考えるんだったなー。。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
if N % 2 == 1:
    NO()
    exit()

XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))

basex1, basey1 = XY[0]
basex2, basey2 = XY[N//2]
for i in range(1, N//2):
    x1, y1 = XY[i]
    x2, y2 = XY[N//2+i]
    # 基準位置からの距離を取る
    dist1 = (basex1-x1, basey1-y1)
    dist2 = (-(basex2-x2), -(basey2-y2))
    if dist1 != dist2:
        NO()
        exit()
YES()
