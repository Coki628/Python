"""
・自力AC！
・4方向から見ていくやつ。今回は累積和取る訳でもなく到達チェックだけなので易しい。
・グリッドが1500^2=225万なので、4方向やるとちょっと重めかとは思ったけどTLE。(制約2.5秒)
　即行C++に写経してACしたのは、終わってみれば悪くない判断だった。
・というのも、終了後結構pypyでみんな通していて、
　試しにゴリさんの提出見たらほぼ同じ方針で0.76秒とかでACしてて、
　いくら何でも2秒近く遅いのはどっか致命的なボトルネックがある、と思って調べたけど、
　なかなか原因が見つからなかったから。
・結局他に似た提出してる人を探すまでして、やっと原因が見つかった。
　boolを全部数字に書き直してた。これをやったらpypyAC0.64秒。
　pypyでboolが遅いなんて話は聞いたことがなかったので、ここで気付けたのはよかったと思う。
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

H, W, N, M = MAP()
grid = list2d(H+2, W+2, 0)
for i in range(H+2):
    grid[i][0] = -1
    grid[i][W+1] = -1
for i in range(W+2):
    grid[0][i] = -1
    grid[H+1][i] = -1

for _ in range(N):
    a, b = MAP()
    grid[a][b] = 1
for _ in range(M):
    c, d = MAP()
    grid[c][d] = -1

ans = list2d(H+2, W+2, 0)
for i in range(1, H+1):
    ok = False
    for j in range(1, W+1):
        if grid[i][j] == 1:
            ok = True
        elif grid[i][j] == -1:
            ok = False
        if ok:
            ans[i][j] = 1
for i in range(1, H+1):
    ok = False
    for j in range(W, 0, -1):
        if grid[i][j] == 1:
            ok = True
        elif grid[i][j] == -1:
            ok = False
        if ok:
            ans[i][j] = 1
for i in range(1, W+1):
    ok = False
    for j in range(1, H+1):
        if grid[j][i] == 1:
            ok = True
        elif grid[j][i] == -1:
            ok = False
        if ok:
            ans[j][i] = 1
for i in range(1, W+1):
    ok = False
    for j in range(H, 0, -1):
        if grid[j][i] == 1:
            ok = True
        elif grid[j][i] == -1:
            ok = False
        if ok:
            ans[j][i] = 1

cnt = 0
for i in range(1, H+1):
    cnt += ans[i].count(1)
print(cnt)
