# -*- coding: utf-8 -*-

"""
・前にpython,pypyで通せなかったやつ、AC。
・pypyのメモリ対策として、DP配列は2次元を1次元2つだけにして、
　100万のグリッドも要素がboolなので2次元目をintのビットで管理した。
・上記ビットの管理は約1000ビットにもなるんだけど、そこはさすが多倍長のpythonだった。
・pypyAC0.5秒。pythonはTLEだったけど、def main()やったらAC0.8秒。
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
INF = float('inf')
MOD = 10 ** 9 + 7

H, W, N = MAP()
# グリッドの要素はboolなので、2次元目はintのビットで持つ
grid = [0] * H
for i in range(H):
    row = LIST()
    for j in range(W):
        if row[j]:
            grid[i] |= 1<<j

# 2次元のDP配列をcur,nxtの2つだけにする
cur = [0] * (W+1)
cur[0] = N - 1
for i in range(H):
    nxt = [0] * (W+1)
    for j in range(W):
        # 通る回数が偶数なら半分ずつ
        if cur[j] % 2 == 0:
            nxt[j] += cur[j] // 2
            cur[j+1] += cur[j] // 2
        else:
            # 通る回数が奇数なら、元のグリッドの状態に応じて1回多く行く方が決まる
            if grid[i] & 1<<j == 0:
                nxt[j] += cur[j] // 2 + 1
                cur[j+1] += cur[j] // 2
            else:
                nxt[j] += cur[j] // 2
                cur[j+1] += cur[j] // 2 + 1
    # N-1回終了時の状態にグリッドを更新
    for j in range(W):
        cur[j] &= 1
        if cur[j]:
            grid[i] ^= 1<<j
    cur = nxt[:]

# N回目の散歩をシミュレーション
i = j = 0
while i < H and j < W:
    if grid[i] & 1<<j == 0:
        i += 1
    else:
        j += 1
print(i+1, j+1)
