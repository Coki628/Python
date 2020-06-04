"""
・これもTLE。
・再帰探索
・グリッドを1次元にしてメモ化再帰にした。けどほとんど変わらず。。
・さらに枝刈りの条件を増やしたりしてみたけど変化なし。。
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
    grid += list(input())

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
    return rank[0]

memo = {}
def rec(grid):
    if tuple(grid) in memo:
        return memo[tuple(grid)]
    done = True
    res = INF
    for i in range(N):
        for j in range(i+1, N):
            if grid[i*N+j] == '-':
                done = False
                if i == 0 or grid[i*N+j] == 'o' and grid[j*N+i] == 'x':
                    grid[i*N+j] = 'o'
                    grid[j*N+i] = 'x'
                    res = min(res, rec(grid))
                elif grid[i*N+j] == 'x' and grid[j*N+i] == 'o':
                    grid[i*N+j] = 'x'
                    grid[j*N+i] = 'o'
                    res = min(res, rec(grid))
                else:
                    grid[i*N+j] = 'o'
                    grid[j*N+i] = 'x'
                    res = min(res, rec(grid))
                    grid[i*N+j] = 'x'
                    grid[j*N+i] = 'o'
                    res = min(res, rec(grid))
                grid[i*N+j] = '-'
                grid[j*N+i] = '-'
    if done:
        res = min(res, check(grid))
    memo[tuple(grid)] = res
    return res

ans = rec(grid)
print(ans)
