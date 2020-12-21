"""
・さくっと自力AC
・累積和
・累積和取っておいて、全箇所で前後比べればOK。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    S = input()
    N = len(S)

    acc = [0] * (N+1)
    for i in range(N):
        acc[i+1] = acc[i] + (S[i] == '1')

    ans = INF
    for i in range(N+1):
        lcnt1 = acc[i] - acc[0]
        lcnt0 = i - lcnt1
        rcnt1 = acc[N] - acc[i]
        rcnt0 = N - i - rcnt1
        ans = min(ans, lcnt0+rcnt1, lcnt1+rcnt0)
    print(ans)
