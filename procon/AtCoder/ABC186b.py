import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**6)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

H, W = MAP()

grid = [[]] * H
for i in range(H):
    grid[i] = LIST()

mn = INF
for i in range(H):
    for j in range(W):
        mn = min(mn, grid[i][j])

ans = 0
for i in range(H):
    for j in range(W):
        ans += grid[i][j] - mn
print(ans)
