"""
・さくっと自力AC
・途中のある位置Xが知りたいので、開始から順操作して終了から逆操作して、直前と直後の比較から判断する。
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

N, K, X = MAP()
X -= 1

P1 = []
P2 = []
for i in range(K):
    if i == X:
        input()
    else:
        a, b = MAP()
        a -= 1; b -= 1
        if i < X:
            P1.append((a, b))
        else:
            P2.append((a, b))

A = list(range(N))
B = [b-1 for b in LIST()]
for a, b in P1:
    A[a], A[b] = A[b], A[a]
for a, b in P2[::-1]:
    B[a], B[b] = B[b], B[a]

ans = []
for i in range(N):
    if A[i] != B[i]:
        ans.append(i+1)
print(*ans)
