"""
・過去の愚行を反省して書き直し。
・ビットで管理したらスッキリ。
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

def popcount(x):
    x -= (x >> 1) & 0x55
    x = (x & 0x33) + ((x >> 2) & 0x33)
    x = (x + (x >> 4)) & 0xf
    return x & 0xf

for _ in range(INT()):
    N = INT()
    grid = [[]] * N
    for i in range(N):
        row = list(input())
        if row[0] == 'S':
            row[0] = -1
        if row[-1] == 'F':
            row[-1] = -1
        grid[i] = list(map(int, row))

    # 変える候補の4箇所
    D = [(0, 1), (1, 0), (N-1, N-2), (N-2, N-1)]
    # 現在の状態
    cur = 0
    for i, (h, w) in enumerate(D):
        cur |= grid[h][w]<<i
    # 良い状態：0011か1100
    good1 = 1<<0 | 1<<1
    good2 = 1<<2 | 1<<3
    ans = []
    # xorで差異を取る
    res1 = cur ^ good1
    res2 = cur ^ good2
    # 0011との差異が2以下ならいけるし、ダメなら1100がいける
    if popcount(res1) <= 2:
        for i, (h, w) in enumerate(D):
            if res1>>i & 1:
                ans.append((h+1, w+1))
    else:
        for i, (h, w) in enumerate(D):
            if res2>>i & 1:
                ans.append((h+1, w+1))

    print(len(ans))
    for a in ans:
        print(*a)
