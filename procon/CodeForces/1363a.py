"""
・なんとか自力AC
・場合分けつらいやつ。
・なんとか通したけど、こどふぉのAこういうのよくあるからしんどいよね。
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
    N, K = MAP()
    A = [a&1 for a in LIST()]

    C = Counter(A)
    if C[1] == 0:
        No()
        continue
    K -= 1
    C[1] -= 1
    if C[0] >= K:
        Yes()
    else:
        if C[0] % 2 == K % 2:
            K -= C[0]
        else:
            if C[0] == 0:
                No()
                continue
            K -= C[0] - 1
        if C[1] >= K:
            Yes()
        else:
            No()
