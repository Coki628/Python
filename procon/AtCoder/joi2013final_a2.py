# -*- coding: utf-8 -*-

"""
・100万グリッド見るだけでpypy1秒間に合わないのはちょっと遅いなーと思って改善。
　(決めればpypyでのfirstACだったのもあって)
・小さいループをif文4つにしたりしても変わらなくて、文字列処理を数値に置き換えたら速くなった。
　pypyTLE→AC0.8秒。(制約1秒)
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

H, W = MAP()
grid = list2d(H+2, W+2, -1)
for i in range(1, H+1):
    row = input()
    for j, s in enumerate(row, 1):
        if s == 'J':
            grid[i][j] = 0
        elif s == 'O':
            grid[i][j] = 1
        else:
            grid[i][j] = 2
        
tmp = ''
tmp += input()
tmp += input()
emb = [0] * 4
for i, s in enumerate(tmp):
    if s == 'O':
        emb[i] = 1
    elif s == 'I':
        emb[i] = 2

# 紋章との一致確認
def check(h, w):
    cur = [grid[h][w], grid[h][w+1], grid[h+1][w], grid[h+1][w+1]]
    return cur == emb

# まず初期状態をカウントしておく
ans = 0
good = list2d(H+2, W+2, 0)
for i in range(1, H):
    for j in range(1, W):
        if check(i, j):
            good[i][j] = 1
            ans += 1

mx = 0
for i in range(1, H+1):
    for j in range(1, W+1):
        for a in range(3):
            if a == grid[i][j]:
                continue
            cnt = 0
            tmp = grid[i][j]
            grid[i][j] = a
            # 結局4箇所しか影響しないので、それぞれどうなるか確認する
            for k in range(-1, 1):
                for l in range(-1, 1):
                    cnt -= good[i+k][j+l]
                    if check(i+k, j+l):
                        cnt += 1
            mx = max(mx, cnt)
            grid[i][j] = tmp
ans += mx
print(ans)
