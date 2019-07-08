# -*- coding: utf-8 -*-

"""
・D自力AC！
・グリッドぐりぐり系
・if条件8方向が冗長だけど、綺麗なの考えるより書く方が早かった。
　→ちょっと考えたら、短い内ループ2つ回せばすぐだった。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

H,W=MAP()
# 四方に一回り大きいグリッドを作る
grid = list2d(H+2, W+2, '*')
for i in range(1, H+1):
    row = list(input())
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

ok=list2d(H+2, W+2, False)
ans=list2d(H+2, W+2, '.')
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j]=='#':
            is_ok=True
            # 自分と8方向の確認
            for k in range(i-1, i+2):
                for l in range(j-1, j+2):
                    if grid[k][l]!='#' and grid[k][l]!='*':
                        is_ok=False
            if is_ok:
                for k in range(i-1, i+2):
                    for l in range(j-1, j+2):
                        ok[k][l]=True
                ans[i][j]='#'
        else:
            ok[i][j]=True

for i in range(1, H+1):
    for j in range(1, W+1):
        if not ok[i][j]:
            print('impossible')
            exit()

print('possible')
for i in range(1, H+1):
    print(''.join(ans[i][1:-1]))
