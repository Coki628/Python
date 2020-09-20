"""
・自力AC
・枝刈り全探索
・Nが100万あるけど、a*bは早々にNよりでかくなるはず、と判断して枝刈りで通した。
・公式解見たけど、やっぱ式変形すればaだけ全探索で(N-1)//aで使えるbの個数出たっぽい。
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

N = INT()
ans = 0
for a in range(1, N+1):
    for b in range(1, N+1):
        if a*b >= N:
            break
        ans += 1
print(ans)
