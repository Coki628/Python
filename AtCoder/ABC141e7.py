# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2019/09/16/014600
　　　https://img.atcoder.jp/abc141/editorial.pdf
・文字列のままのやつに、ロリハと同じ高速化を適用。
・若干遅いけどこっちでも十分な速さでAC。python0.1秒、pypy0.5秒。
　こうなるとハッシュ化する意味薄いなー。。
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
    A = {}
    for i in range(N-m+1):
        a = S[i:i+m]
        if a in A:
            # aが2回目以降に出現したindex
            j = A[a]
            # iとjがm以上離れているかどうか
            if i-j >= m:
                return True
        else:
            # aが最初に出現したindex
            A[a] = i
    return False

print(bisearch_max(0, N, calc))
