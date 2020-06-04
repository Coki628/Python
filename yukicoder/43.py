"""
・これはTLE。
・再帰探索
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
grid = [[]] * N
for i in range(N):
    grid[i] = list(input())

def check(grid):
    win = [0] * N
    for i in range(N):
        for j in range(i+1, N):
            if grid[i][j] == 'o':
                win[i] += 1
            elif grid[i][j] == 'x':
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
    return rank[0]

def rec(grid):
    done = True
    res = INF
    for i in range(N):
        for j in range(i+1, N):
            if grid[i][j] == '-':
                done = False
                grid[i][j] = 'o'
                grid[j][i] = 'x'
                res = min(res, rec(grid))
                if i != 0:
                    grid[i][j] = 'x'
                    grid[j][i] = 'o'
                    res = min(res, rec(grid))
                grid[i][j] = '-'
                grid[j][i] = '-'
    if done:
        res = min(res, check(grid))
    return res

ans = rec(grid)
print(ans)
