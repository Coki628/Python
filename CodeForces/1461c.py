"""
・確率、余事象、累積積
・多分誤差で落ちてる。C++でACしたので、色々コメントもそっちで。
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
    N, M = MAP()
    A = LIST()

    P = [1] * N
    for i in range(N):
        if A[i] == i+1:
            P[i] = -1

    acc = [1] * (N+1)
    done = [0] * (N+1)
    for i in range(M):
        r, p = input().split()
        r = int(r)
        p = float(p)
        if 1-p == 0:
            done[0] += 1
            done[r] -= 1
        else:
            acc[0] *= 1 - p
            acc[r] /= 1 - p
    for i in range(N):
        acc[i+1] *= acc[i]
        done[i+1] += done[i]

    for i in range(N):
        if done[i]:
            P[i] = 1
        elif P[i] != -1:
            P[i] *= acc[i]

    for i in range(N):
        if done[i]:
            continue
        if P[i] != -1:
            P[i] = 1 - P[i]
    ans = 1
    for p in P:
        if p != -1:
            ans = min(ans, p)
    print(ans)
