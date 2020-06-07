"""
・自力AC
・XOR系、ビット全探索
・Nが1024、値も1024までしか取らないので、ありえるパターンを全部試せる。
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
    N = INT()
    A = LIST()

    M = 0
    for a in A:
        M = max(M, a.bit_length())
    ans = INF
    for bit in range(1, 1<<M):
        B = [0] * N
        for i in range(N):
            B[i] = A[i] ^ bit
        if set(A) == set(B):
            ans = min(ans, bit)
    if ans == INF:
        print(-1)
    else:
        print(ans)
