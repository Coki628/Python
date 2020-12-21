"""
・ここまで速攻自力AC
・ダブリング
・周期でうまくやる解法もあったぽいけど、真っ先にダブリング思ったので突き進んだ。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, K = MAP()
A = [a-1 for a in LIST()]

MAX = 64
nxt = list2d(MAX, N, -1)
for i, a in enumerate(A):
    nxt[0][i] = a

for k in range(1, MAX):
    for i in range(N):
        nxt[k][i] = nxt[k-1][nxt[k-1][i]]

ans = 0
for k in range(MAX-1, -1, -1):
    if K >> k & 1:
        ans = nxt[k][ans]
print(ans+1)
