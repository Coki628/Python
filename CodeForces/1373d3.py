"""
・区間和の最大値
・累積min,maxで線形もやってみた。セグ木と大差なかったけどpypyAC0.50→0.42秒。
・さすがにlog乗らない訳だし、TL厳しい時はこっち使った方がいいと思う。
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
    N = INT()
    A = LIST()

    B1 = []
    B2 = []
    for i in range(N-1):
        if i % 2 == 0:
            B1.append(A[i+1]-A[i])
        else:
            B2.append(A[i]-A[i+1])
    acc1 = [0] + list(accumulate(B1))
    acc2 = [0] + list(accumulate(B2))
    # 前から累積minで一番いい左端を集める
    accmn1 = list(accumulate(acc1, min))
    # 後ろから累積maxで一番いい右端を集める
    accmx1 = list(accumulate(acc1[::-1], max))[::-1]
    accmn2 = list(accumulate(acc2, min))
    accmx2 = list(accumulate(acc2[::-1], max))[::-1]
    N1 = len(acc1)
    N2 = len(acc2)

    base = sum([a if i % 2 == 0 else 0 for i, a in enumerate(A)])
    ans = base
    for i in range(N1-1):
        l = accmn1[i]
        r = accmx1[i+1]
        ans = max(ans, base+r-l)
    for i in range(N2-1):
        l = accmn2[i]
        r = accmx2[i+1]
        ans = max(ans, base+r-l)
    print(ans)
