"""
・自分が途中までやろうとしてた方針。実はちょっと直したらこれでも通った。。
・二分探索+境界付近全探索
・p,qの大小で場合分けしないといけないのは同じ。(それによってOK範囲が大小どっちに寄るか変わるから)
・p==qは自明場合分け。p>qなら小さい方にOKが寄るので最大値を探す。p<qなら大きい方にOKが寄るので最小値を探す。
・実はここまでできていて、微妙な端数調整の+1とか直したら通ったっていうね。。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def bisearch_max(mn, mx, func):
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

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

def check(x):
    a = (100+p) * x // 100
    b = (100+q) * x // 100 + r
    return a < b

p, q, r = MAP()
MX = 10**9

if p == q:
    if r == 0:
        print(0)
    else:
        print(MX)
elif p > q:
    res = bisearch_max(0, MX+1, check)
    lo = max(res-100000, 1)
    hi = min(res+100000, MX+1)
    ans = lo - 1
    for i in range(lo, hi):
        if check(i):
            ans += 1
    print(ans)
else:
    res = bisearch_min(0, MX+1, check)
    lo = max(res-100000, 1)
    hi = min(res+100000, MX+1)
    ans = MX + 1 - hi
    for i in range(lo, hi):
        if check(i):
            ans += 1
    print(ans)
