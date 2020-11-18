"""
参考：https://twitter.com/mugen_1337/status/1328743053448777728
・自力ならず。。
・地獄実装
・2*2マスは確実に4手以内で処理できるので、基本はそれをやって奇数の端は別処理。
　その方針は決して間違ってはなかったんだけど詰めきれず。。
・先にメイン処理して、最後に端の例外処理してたんだけど、これだと最悪ケースでは回数オーバー。
　先に端を潰して、後で残った中身部分をやったらよかった。。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def check(i, j):
    res = []
    cnt = grid[i][j] + grid[i][j+1] + grid[i+1][j] + grid[i+1][j+1]
    if cnt == 1:
        if grid[i][j]:
            res.append((i+1, j+1, i+1, j+2, i+2, j+1))
            res.append((i+1, j+1, i+1, j+2, i+2, j+2))
            res.append((i+1, j+1, i+2, j+1, i+2, j+2))
        elif grid[i+1][j]:
            res.append((i+2, j+1, i+1, j+2, i+1, j+1))
            res.append((i+2, j+1, i+1, j+2, i+2, j+2))
            res.append((i+2, j+1, i+1, j+1, i+2, j+2))
        elif grid[i][j+1]:
            res.append((i+1, j+2, i+1, j+1, i+2, j+1))
            res.append((i+1, j+2, i+1, j+1, i+2, j+2))
            res.append((i+1, j+2, i+2, j+1, i+2, j+2))
        elif grid[i+1][j+1]:
            res.append((i+2, j+2, i+1, j+2, i+2, j+1))
            res.append((i+2, j+2, i+1, j+2, i+1, j+1))
            res.append((i+2, j+2, i+2, j+1, i+1, j+1))
    elif cnt == 2:
        # 斜め
        if grid[i][j] and grid[i+1][j+1]:
            res.append((i+1, j+1, i+1, j+2, i+2, j+1))
            res.append((i+2, j+2, i+1, j+2, i+2, j+1))
        elif grid[i+1][j] and grid[i][j+1]:
            res.append((i+1, j+1, i+1, j+2, i+2, j+2))
            res.append((i+1, j+1, i+2, j+1, i+2, j+2))
        # 縦
        elif grid[i][j] and grid[i+1][j]:
            res.append((i+1, j+1, i+1, j+2, i+2, j+2))
            res.append((i+2, j+1, i+1, j+2, i+2, j+2))
        elif grid[i][j+1] and grid[i+1][j+1]:
            res.append((i+1, j+2, i+1, j+1, i+2, j+1))
            res.append((i+2, j+2, i+1, j+1, i+2, j+1))
        # 横
        elif grid[i][j] and grid[i][j+1]:
            res.append((i+1, j+1, i+2, j+1, i+2, j+2))
            res.append((i+1, j+2, i+2, j+1, i+2, j+2))
        elif grid[i+1][j] and grid[i+1][j+1]:
            res.append((i+2, j+2, i+1, j+2, i+1, j+1))
            res.append((i+2, j+1, i+1, j+2, i+1, j+1))
    elif cnt == 3:
        if grid[i][j] and grid[i][j+1] and grid[i+1][j+1]:
            res.append((i+1, j+1, i+1, j+2, i+2, j+2))
        elif grid[i][j+1] and grid[i+1][j+1] and grid[i+1][j]:
            res.append((i+1, j+2, i+2, j+2, i+2, j+1))
        elif grid[i+1][j+1] and grid[i+1][j] and grid[i][j]:
            res.append((i+2, j+2, i+2, j+1, i+1, j+1))
        elif grid[i+1][j] and grid[i][j] and grid[i][j+1]:
            res.append((i+2, j+1, i+1, j+1, i+1, j+2))
    elif cnt == 4:
        res.append((i+1, j+1, i+1, j+2, i+2, j+1))
        res.append((i+1, j+1, i+1, j+2, i+2, j+2))
        res.append((i+1, j+1, i+2, j+1, i+2, j+2))
        res.append((i+1, j+2, i+2, j+1, i+2, j+2))
    grid[i][j] = grid[i][j+1] = grid[i+1][j] = grid[i+1][j+1] = 0
    return res

def checkh(i, j):
    res = []
    if grid[i][j] and grid[i+1][j]:
        res.append((i+1, j+1, i+1, j+0, i+2, j+0))
        res.append((i+2, j+1, i+1, j+0, i+2, j+0))
    elif grid[i][j]:
        res.append((i+1, j+1, i+1, j+0, i+2, j+0))
        grid[i][j-1] ^= 1
        grid[i+1][j-1] ^= 1
    elif grid[i+1][j]:
        res.append((i+2, j+1, i+1, j+0, i+2, j+0))
        grid[i][j-1] ^= 1
        grid[i+1][j-1] ^= 1
    grid[i][j] = grid[i+1][j] = 0
    return res

def checkw(i, j):
    res = []
    if grid[i][j] and grid[i][j+1]:
        res.append((i+1, j+1, i+0, j+1, i+0, j+2))
        res.append((i+1, j+2, i+0, j+1, i+0, j+2))
    elif grid[i][j]:
        res.append((i+1, j+1, i+0, j+1, i+0, j+2))
        grid[i-1][j] ^= 1
        grid[i-1][j+1] ^= 1
    elif grid[i][j+1]:
        res.append((i+1, j+2, i+0, j+1, i+0, j+2))
        grid[i-1][j] ^= 1
        grid[i-1][j+1] ^= 1
    grid[i][j] = grid[i][j+1] = 0
    return res

for _ in range(INT()):
    H, W = MAP()
    grid = [[]] * H
    for i in range(H):
        grid[i] = list(map(int, input()))

    ans = []
    if H % 2 == 1:
        for j in range(W-1):
            ans += checkw(H-1, j)
    if W % 2 == 1:
        for i in range(H-1):
            ans += checkh(i, W-1)
    for i in range(0, H-H%2, 2):
        for j in range(0, W-W%2, 2):
            ans += check(i, j)

    print(len(ans))
    for a in ans:
        print(*a)
