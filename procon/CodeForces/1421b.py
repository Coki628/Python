"""
・地獄場合分け
・絶対もっと綺麗にできるやつ。そして条件1箇所書き違えてシステス落ちした。。つら。。
・これはもう条件修正してACさせたやつ。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
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

for _ in range(INT()):
    N = INT()
    grid = [[]] * N
    for i in range(N):
        grid[i] = list(input())

    s1 = grid[0][1]
    s2 = grid[1][0]
    g1 = grid[N-1][N-2]
    g2 = grid[N-2][N-1]

    ans = []
    if s1 == s2 == '0':
        if g1 == '0':
            ans.append((N, N-1))
        if g2 == '0':
            ans.append((N-1, N))
    elif s1 == '1' and s2 == '0':
        if g1 == g2 == '1':
            ans.append((1, 2))
        elif g1 == g2 == '0':
            ans.append((2, 1))
        elif g1 == '1' and g2 == '0':
            ans.append((2, 1))
            ans.append((N, N-1))
        elif g1 == '0' and g2 == '1':
            ans.append((2, 1))
            ans.append((N-1, N))
    elif s1 == '0' and s2 == '1':
        if g1 == g2 == '1':
            ans.append((2, 1))
        elif g1 == g2 == '0':
            ans.append((1, 2))
        elif g1 == '1' and g2 == '0':
            ans.append((1, 2))
            ans.append((N, N-1))
        elif g1 == '0' and g2 == '1':
            ans.append((1, 2))
            ans.append((N-1, N))
    else:
        if g1 == '1':
            ans.append((N, N-1))
        if g2 == '1':
            ans.append((N-1, N))

    print(len(ans))
    for a in ans:
        print(*a)
