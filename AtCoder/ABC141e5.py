# -*- coding: utf-8 -*-

"""
・自力TLEだった普通に文字列探すやつを元に変更
・今までで1番速くなったけど、2ケースだけTLE。
・さらにちょっと修正。状況は変わらず。
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

N = INT()
S = input()

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """

    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

def calc(m):
    A = set()
    for i in range(N-m*2+1):
        a = S[i:i+m]
        # 既に探したaはもう1回探す必要ない
        if a not in A:
            b = S[i+m:]
            if b.find(a) != -1:
                return True
            A.add(a)
    return False

print(bisearch_max(0, N, calc))
