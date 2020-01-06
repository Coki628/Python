# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC！
・切り方2つを縦横で計4通り全部試す。
・T字の横棒はH,Wが10^5なので全探索OK。
・的確に必要なパターンを出してまとめられたから良かった。
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

H, W = MAP()

# 1方向で3つに切る
def check1(h, w):
    if h < 3:
        return INF
    if h % 3 == 0:
        return 0
    else:
        return w

# T字に切る
def check2(h, w):
    if h < 2 or w < 2:
        return INF
    res = INF
    # Tの横棒をどの位置にするか全部試す
    for i in range(1, h):
        a = i * w
        b = (h-i) * (w // 2)
        c = (h-i) * ceil(w, 2)
        res = min(res, max(a, b, c) - min(a, b, c))
    return res

# 2種類の切り方を縦横に試してみて一番小さくなるものを使う
ans = min(check1(H, W), check1(W, H), check2(H, W), check2(W, H))
print(ans)
