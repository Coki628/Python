# -*- coding: utf-8 -*-

"""
・自力AC！
・累積和、MODでまとめる。
"""

import sys
from math import gcd
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

def lcm(x, y): return (x * y) // gcd(x, y)

for _ in range(INT()):
    mod1, mod2, Q = MAP()

    # mod1とmod2のLCMで周期を作る
    N = lcm(mod1, mod2)
    ok = [0] * N
    for i in range(N):
        # 条件に合うところを確認
        if i%mod1%mod2 != i%mod2%mod1:
            ok[i] = 1
    # 区間を取れるように累積和しておく
    acc = [0] + list(accumulate(ok))

    ans = [0] * Q
    for i in range(Q):
        l, r = MAP()
        # 区間全体をまとめて足す分
        ln = l // N
        rn = r // N
        segcnt = rn - ln
        # 左端と右端の周辺
        modl = l % N
        modr = r % N
        # 右端がある区間の右端以前は足して、左端がある区間の左端以前は引く
        cnt = acc[modr+1] - acc[modl]
        ans[i] = segcnt * acc[N] + cnt
    print(*ans)
