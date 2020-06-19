"""
・グリッド、構築
・題意を満たす形は斜めに繋げてく感じ。(自分ノート参照)
・色々ガチャガチャやってたらいいのが降ってきたからよかったけど、こういうの思いつかないとつらいよね。
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

N = INT()
grid = list2d(N+2, N+2, 0)
directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

for i in range(1, N+1):
    grid[i][i] = 1
    for di, dj in directions:
        ii = i + di
        ij = i + dj
        grid[ii][ij] = 1
grid[0][0] = grid[N+1][N+1] = 1

ans = []
for i in range(N+2):
    for j in range(N+2):
        if grid[i][j]:
            ans.append((i, j))

print(len(ans))
for i, j in ans:
    print(i, j)
