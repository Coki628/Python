"""
参考：https://codeforces.com/contest/1371/submission/85685702
・自力ならず。。結構考えたのに。
・グリッド、構築
・斜めに行かないとダメ、はやった。けどなんか間隔空けなきゃいけないような気がしてしまって、
　格子状っぽく構築しようとしてた。
・実際は斜めに順番に普通に進んで、はみ出す分はmod N周期で反対側埋めに行けばよかった。
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
    N, K = MAP()

    grid = list2d(N, N, 0)
    cnt = K
    for i in range(N):
        if not cnt:
            break
        for j in range(N):
            if not cnt:
                break
            grid[j][(j+i)%N] = 1
            cnt -= 1

    mnh = mnw = INF
    mxh = mxw = 0
    for i in range(N):
        mnw = min(mnw, grid[i].count(1))
        mxw = max(mxw, grid[i].count(1))
    grid = list(zip(*grid))
    for i in range(N):
        mnh = min(mnh, grid[i].count(1))
        mxh = max(mxh, grid[i].count(1))
    ans = (mxh-mnh) + (mxw-mnw)
    print(ans)
    grid = list(zip(*grid))
    for i in range(N):
        print(''.join(map(str, grid[i])))
