# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/72950
・公式解
・全体の総和以下にできるかどうかは、実は前後から累積和で0以下があるかだけで分かる。
・なんか前後からだけだと、直感的に真ん中らへんの区間和でうまくいくケースを取れない
　気がしてしまうんだけど、前後からだけで大丈夫なんだねー。。
"""

import sys
from itertools import accumulate

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

for _ in range(INT()):
    N = INT()
    A = LIST()

    acc = list(accumulate(A))
    accrev = list(accumulate(A[::-1]))
    if min(acc) <= 0 or min(accrev) <= 0:
        NO()
    else:
        YES()
