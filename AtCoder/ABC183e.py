"""
・自力AC！
・貰うDP、グリッドDP、累積和
・3方向の累積和を取りながらグリッドDPする。
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

def build_grid(H, W, intv, _type, space=False, padding=True):
    # 入力がスペース区切りかどうか
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    # 余白の有無
    if padding:
        offset = 1
    else:
        offset = 0
    grid = list2d(H+offset*2, W+offset*2, intv)
    for i in range(offset, H+offset):
        row = _list()
        for j in range(offset, W+offset):
            grid[i][j] = row[j-offset]
    return grid

H, W = MAP()
grid = build_grid(H, W, '#', str)

dp = list2d(H+1, W+1, 0)
acc1 = list2d(H+1, W+1, 0)
acc2 = list2d(H+1, W+1, 0)
acc3 = list2d(H+1, W+1, 0)
dp[1][1] = 1
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j] == '.':
            dp[i][j] += acc1[i-1][j] + acc2[i][j-1] + acc3[i-1][j-1]
            dp[i][j] %= MOD
            acc1[i][j] += dp[i][j] + acc1[i-1][j]
            acc2[i][j] += dp[i][j] + acc2[i][j-1]
            acc3[i][j] += dp[i][j] + acc3[i-1][j-1]
            acc1[i][j] %= MOD
            acc2[i][j] %= MOD
            acc3[i][j] %= MOD
ans = dp[H][W]
print(ans)
