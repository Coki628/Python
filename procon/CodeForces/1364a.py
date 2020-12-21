"""
・自力AC
・累積和、zero sum ranges系
・必要なのが「xの倍数ではない最長区間」なので、広い方から見ていく。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, x = MAP()
    A = LIST()

    acc = [0] + list(accumulate(A))
    acc = [a%x for a in acc]
    ans = -1
    done = set()
    for l in range(N):
        # 同じ値は2回見る必要ない
        if acc[l] in done:
            continue
        done.add(acc[l])
        for r in range(N, l, -1):
            # 左端と右端が異なれば、区間和はxの倍数じゃない
            if acc[l] != acc[r]:
                ans = max(ans, r-l)
                break
    print(ans)
