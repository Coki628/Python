"""
・自力AC。
・桁和、ビット全探索
・3の倍数判定は桁和が3の倍数になるかどうか。これは覚えてた。
・今回は桁が18桁しかないので、使うかどうかのビット全探索でOK。
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

def digit_sum(n):
    """ 桁和：O(logN) """

    ans = 0
    while n > 0:
        ans += n % 10
        n //= 10
    return ans

def popcount(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

S = input()
N = len(S)

ans = INF
for bit in range(1, 1<<N):
    num = ''
    for i in range(N):
        if bit & (1<<i):
            num += S[i]
    if digit_sum(int(num)) % 3 == 0:
        ans = min(ans, N-popcount(bit))
if ans == INF:
    print(-1)
else:
    print(ans)
