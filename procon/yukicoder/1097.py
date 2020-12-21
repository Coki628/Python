"""
・なんとか自力AC。。
・ダブリング
・解けてみればダブリング典型なのに、時間かかってしまった。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def doubling(MAX, A):
    """ ダブリング """

    N = len(A)
    nxt = list2d(MAX, N, -1)
    sm = list2d(MAX, N, 0)
    for i, a in enumerate(A):
        # 行き先
        nxt[0][i] = (i+a) % N
        # そこまでの和
        sm[0][i] = a
    for k in range(1, MAX):
        for i in range(N):
            nxt[k][i] = nxt[k-1][nxt[k-1][i]]
            sm[k][i] = sm[k-1][i] + sm[k-1][nxt[k-1][i]]
    return (nxt, sm)

N = INT()
A = LIST()

MX = 40
nxt, sm = doubling(MX, A)

# x = cur = 0
# for i in range(10):
#     x += sm[0][cur]
#     cur = nxt[0][cur] % N
#     print(x, cur)

for _ in range(INT()):
    k = INT()
    x = cur = 0
    for i in range(MX-1, -1, -1):
        if k>>i & 1:
            x += sm[i][cur]
            cur = nxt[i][cur]
    print(x)
