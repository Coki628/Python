"""
・自力AC
・制約を見ると、N^2いけるのでどうするか考える。
・全ペア同値にしたいので、その同値をいくつにするか全探索。
・その中で、データ持ち替えて集計したやつ使って、なるべく組めるとこ組ませればいい。
"""

import sys
from collections import Counter

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

    ans = 0
    for i in range(1, N*2+1):
        used = [0] * N
        C = Counter(A)
        cnt = 0
        for j, a in enumerate(A):
            if a == i-a:
                if C[a] >= 2:
                    C[a] -= 2
                    cnt += 1
            else:
                if C[a] and C[i-a]:
                    C[a] -= 1
                    C[i-a] -= 1
                    cnt += 1
        ans = max(ans, cnt)
    print(ans)
