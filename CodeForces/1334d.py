# -*- coding: utf-8 -*-

"""
・自力WA。
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

def gen_arr(v, n):
    res = [0] * n
    x = 2
    for i in range(n-1):
        if i % 2 == 0:
            res[i] = v
        else:
            res[i] = x
            x += 1
    res[-1] = 1
    return res

for _ in range(INT()):
    N, l, r = MAP()
    l -= 1

    tmp = [1, 2, 1]
    ans = []
    if l < 3:
        ans += tmp[l:r]
        vcnt = 3
        incr = 4
        ln = r - l
        while len(ans) < ln:
            ans += gen_arr(vcnt, incr)
            vcnt += 1
            incr += 2
        ans = ans[:ln]
    else:
        cur = 3
        vcnt = 3
        incr = 4
        while cur + incr < l:
            cur += incr
            vcnt += 1
            incr += 2
        ans = gen_arr(vcnt, incr)
        ans = ans[l-cur:]
        ln = r - l
        vcnt += 1
        incr += 2
        while len(ans) < ln:
            ans += gen_arr(vcnt, incr)
            vcnt += 1
            incr += 2
        ans = ans[:ln]
    print(*ans)
