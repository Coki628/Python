"""
・自力AC！
・ダブリング
・内容を読むと、結局値はmod Mの範囲内でしか遷移しないのが分かるので、これは周期になる。
・答えに欲しいのはそこまでの和なので、次の場所と合わせて、和についてもダブリングを用意する。
・後は初期位置の分最初からXにして、その分移動回数N-1にしたり、微調整はあるけどまあ普通にやる。
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
        nxt[0][i] = a
        sm[0][i] = a
    for k in range(1, MAX):
        for i in range(N):
            nxt[k][i] = nxt[k-1][nxt[k-1][i]]
            sm[k][i] = sm[k-1][i] + sm[k-1][nxt[k-1][i]]
    return (nxt, sm)

N, X, M = MAP()

MAX = 40
A = [0] * M
for i in range(M):
    A[i] = pow(i, 2, M)
nxt, sm = doubling(MAX, A)

cur = X
ans = X
N -= 1
for k in range(MAX-1, -1, -1):
    if N >> k & 1:
        ans += sm[k][cur]
        cur = nxt[k][cur]
print(ans)
