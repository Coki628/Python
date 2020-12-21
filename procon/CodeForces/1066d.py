# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/62419
・自力ならず。問題を汲み取れなかった。。
・二分探索
・1回全部入りきらないが確定したら、操作を全部最初からにリセットする。(ここが分かってなかった。)
　なんか最低限の箱だけ空にして最適になるように残りを頑張って詰めるとかそんなんかと思った。
・話が理解できたら後はにぶたんで各開始位置から試すだけ。
・計算量が20万のNにlogだけど、意外とギリギリ0.97秒。(制約1秒)
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def bisearch_min(mn, mx, func):
    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

N, M, K = MAP()
A = LIST()

def check(m):
    box = [0] * M
    i = m
    j = 0
    while i < N:
        if box[j] + A[i] <= K:
            box[j] += A[i]
            i += 1
        else:
            j += 1
            if j == M:
                return False
    return True

res = bisearch_min(-1, N, check)
ans = N - res
print(ans)
