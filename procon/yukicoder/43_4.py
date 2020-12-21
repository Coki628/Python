"""
・やっと自力AC。。
・ビット全探索
・再帰で埒が明かなかったので方針転換。グリッド位置とビットの紐付けやるのめんどいなーって
　この方針は後回しにしてたんだけど、さっさとこっちでやればよかった。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N = INT()
grid = []
for i in range(N):
    # グリッドは1次元で
    grid += list(input())

# 試合結果の集計と順位確定
def check(grid):
    win = [0] * N
    for i in range(N):
        for j in range(i+1, N):
            if grid[i*N+j] == 'o':
                win[i] += 1
            elif grid[i*N+j] == 'x':
                win[j] += 1
    win = [(a, i) for i, a in enumerate(win)]
    win.sort(reverse=1)
    rank = [0] * N
    cur = 1
    rank[win[0][1]] = cur
    for i in range(1, N):
        if win[i][0] != win[i-1][0]:
            cur += 1
        rank[win[i][1]] = cur
    # チーム0の順位を返す
    return rank[0]

# グリッド位置とビットの紐付け
gtob = {}
b = 0
for i in range(N):
    for j in range(i+1, N):
        gtob[i*N+j] = b
        b += 1

ans = INF
for bit in range(1<<b):
    cur = grid[:]
    for i in range(N):
        for j in range(i+1, N):
            # この位置が未確定なら
            if cur[i*N+j] == '-':
                # ビットの状態に応じて勝敗を確定する
                k = gtob[i*N+j]
                if bit>>k & 1:
                    cur[i*N+j] = 'o'
                    cur[j*N+i] = 'x'
                else:
                    cur[i*N+j] = 'x'
                    cur[j*N+i] = 'o'
    ans = min(ans, check(cur))
print(ans)
