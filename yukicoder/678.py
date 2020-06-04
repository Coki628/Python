"""
・自力AC
・実装おつ系
・最初、例1の図で横にビームがかすった奴もやられるのかと思ってしまって、
　相当めんどくさい実装をやろうとしていた。普通に上向きにしか当たり判定ないって
　気づくまでにすげーWA出しまくって時間かかった。。
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
EPS = 10 ** -10

N, lb, rb = MAP()
H, W = 1680, 1280
grid = list2d(H+1, W+1, -1)

for i in range(N):
    xl, yu, xr, yd = MAP()
    xl = max(xl, 0)
    xr = min(xr, W)
    yu = max(yu, 0)
    yd = min(yd, H)
    for j in range(xl, xr+1):
        grid[yd][j] = i

ans = [0] * N
for j in range(lb, rb+1):
    for i in range(H, -1, -1):
        if grid[i][j] != -1:
            ans[grid[i][j]] = 1
            break
for a in ans:
    print(a)
