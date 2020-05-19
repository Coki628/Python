"""
参考：https://www.youtube.com/watch?v=MXOSqPzsiqo&feature=youtu.be
・終了後、時間をかけて粘ったけど方針知ってても自力実装ACならず。。
・グリッドBFS、座標圧縮応用
・グリッドを2倍にする方針はやろうとしたんだけど、元の値を2倍したやつを座圧してたらバグって死んだ。。
・正解は、座圧した値をグリッドに展開する時に2倍にする、だった。
　これだと、偶数(を含む)マスは全部壁用みたいになるから、奇数マスだけ加算対象みたいになってうまくいく。
・これで想定解だと思うけど、最大3000^2を2倍にした6000^2=3600万グリッドがきつい。制約3秒だけどpypyTLE。。
・C++でAC確認。
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

def compress(S):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

def bfs(grid, src):
    """ BFS(グリッド、重みなし) """
    from collections import deque

    H, W = len(grid), len(grid[0])
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    visited = list2d(H, W, 0)
    que = deque()
    for h, w in src:
        que.append((h, w))
        visited[h][w] = 1
    while que:
        h, w = que.popleft()
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            # 外壁まで行ったら即終了
            if grid[h2][w2] == -1:
                print('INF')
                exit()
            if grid[h2][w2] == 1:
                continue
            if visited[h2][w2]:
                continue
            visited[h2][w2] = 1
            que.append((h2, w2))
    return visited

N, M = MAP()
vert = []
horz = []
S1 = set([0])
S2 = set([0])
for i in range(N):
    h1, h2, w = MAP()
    vert.append((h1, h2, w))
    S1.add(h1)
    S1.add(h2)
    S2.add(w)
for i in range(M):
    h, w1, w2 = MAP()
    horz.append((h, w1, w2))
    S1.add(h)
    S2.add(w1)
    S2.add(w2)

zph, unzph = compress(S1)
zpw, unzpw = compress(S2)
# グリッドの壁配置は全部2倍して考える
H, W = len(zph) * 2, len(zpw) * 2

grid = list2d(H, W, -1)
for i in range(1, H-1):
    for j in range(1, W-1):
        grid[i][j] = 0
for i in range(N):
    h1, h2, w = vert[i]
    w = zpw[w] * 2
    for h in range(zph[h1]*2, zph[h2]*2+1):
        grid[h][w] = 1
for i in range(M):
    h, w1, w2 = horz[i]
    h = zph[h] * 2
    for w in range(zpw[w1]*2, zpw[w2]*2+1):
        grid[h][w] = 1

res = bfs(grid, [(zph[0]*2, zpw[0]*2)])
# for row in res:
#     print(row)

ans = 0
# 対象は奇数マスだけ
for i in range(1, H-1, 2):
    for j in range(1, W-1, 2):
        if res[i][j]:
            hln = unzph[i//2+1] - unzph[i//2]
            wln = unzpw[j//2+1] - unzpw[j//2]
            ans += hln * wln
print(ans)
