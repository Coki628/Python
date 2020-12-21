"""
・なんとか自力AC。
・小数の二分探索
・最初、交差する直前まで区間を進めて、最後の詰めは算数、の方針やろうとしたけどうまくいかず。
・ひと呼吸して、方針考え直したら、あ、これにぶたん行けるんじゃあ。。ってなった。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def bisearch_min(mn, mx, func, times):
    ok = mx
    ng = mn
    for _ in range(times):
        mid = (ok+ng) / 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

def check(m):
    cur = 0
    sp = 1
    loc1 = INF
    for i in range(1, N+2):
        dist = A[i] - A[i-1]
        t = dist / sp
        if cur + t > m:
            loc1 = A[i-1] + sp*(m-cur)
            break
        cur += t
        sp += 1
    
    cur = 0
    sp = 1
    loc2 = -INF
    for i in range(N, -1, -1):
        dist = A[i+1] - A[i]
        t = dist / sp
        if cur + t > m:
            loc2 = A[i+1] - sp*(m-cur)
            break
        cur += t
        sp += 1

    return loc1 >= loc2

for _ in range(INT()):
    N, L = MAP()
    A = [0] + LIST() + [L]

    res = bisearch_min(0, INF, check, 100)
    print(res)
