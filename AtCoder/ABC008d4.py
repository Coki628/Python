# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-8
・分割統治法、再帰探索
・概要覚えてたのに、+1するかしないかとか1回目の処理をどうするかとかで
　色々悩んだりバグらせたりで時間かかった。。
・まあこれは部分点解法のようでTLE
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

W, H = MAP()
N = INT()
XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))

def rec(x1, y1, x2, y2):
    res = 0
    for x3, y3 in XY:
        if x1 <= x3 <= x2 and y1 <= y3 <= y2:
            tmp = (x2-x1+1) + (y2-y1+1) - 1
            # 左下
            tmp += rec(x1, y1, x3-1, y3-1)
            # 右上
            tmp += rec(x3+1, y3+1, x2, y2)
            # 左上
            tmp += rec(x1, y3+1, x3-1, y2)
            # 右下
            tmp += rec(x3+1, y1, x2, y3-1)
            res = max(res, tmp)
    return res

ans = 0
for x, y in XY:
    tmp = W + H - 1
    # 左下
    tmp += rec(1, 1, x-1, y-1)
    # 右上
    tmp += rec(x+1, y+1, W, H)
    # 左上
    tmp += rec(1, y+1, x-1, H)
    # 右下
    tmp += rec(x+1, 1, W, y-1)
    ans = max(ans, tmp)
print(ans)
