# -*- coding: utf-8 -*-

"""
・終了後も色々頑張ったけどこの方針だと無理だった。
・にぶたんで近い所見つけて周辺探しても、まだ足りない場合がある。
"""

import sys
from collections import Counter
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
# sys.setrecursionlimit(10 ** 9)
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

def calc1(x):
    cnt = [0] * 2
    cnt[0] = C['0'] * x
    cnt[1] = C['1'] * x
    return cnt[0] - cnt[1]

def calc2(x):
    d, mo = divmod(x, N)
    cnt = [0] * 2
    cnt[0] = C['0'] * d
    cnt[1] = C['1'] * d
    return cnt[0] - cnt[1] + acc[mo]

for _ in range(INT()):

    N, X = MAP()
    S = input()
    C = Counter(S)

    acc = [1 if c == '0' else -1 for c in S]
    acc = [0] + list(accumulate(acc))

    if C['0'] == C['1']:
        for i in range(N):
            if calc2(i) == X:
                print(-1)
                break
        else:
            print(0)
    elif X < 0 and C['0'] > C['1'] or X > 0 and C['0'] < C['1']:
        cnt = 0
        for i in range(N):
            if calc2(i) == X:
                cnt += 1
        print(cnt)
    else:
        res = bisearch_min(0, 10**18, lambda m: abs(calc1(m)) >= abs(X))
        cnt = 0
        for i in range(max(res-10, 0), res+11):
            for j in range(N):
                if abs(calc2(i*N+j)) == abs(X):
                    cnt += 1
        print(cnt)
