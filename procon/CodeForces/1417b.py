"""
・基本はペアになるところを反対側に置く。同値ペアになってしまうところだけ半分に分ける。
"""

import sys
from collections import Counter

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

for _ in range(INT()):
    N, T = MAP()
    A = LIST()

    C = Counter(A)
    D = {}
    ans = [-1] * N
    for k1, v in C.items():
        if k1 in D:
            continue
        k2 = T - k1
        if k1 == k2:
            cnt = v // 2
            for i in range(N):
                if A[i] == k1:
                    if cnt:
                        ans[i] = 0
                        cnt -= 1
                    else:
                        ans[i] = 1
        else:
            D[k1] = 0
            D[k2] = 1
    for i in range(N):
        if ans[i] == -1:
            ans[i] = D[A[i]]
    print(*ans)
