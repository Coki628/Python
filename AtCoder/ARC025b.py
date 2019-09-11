# -*- coding: utf-8 -*-

"""
・自力AC！
・二次元累積和、白黒分離させて考える。
・O((HW)^2)=10000^2=1億は無理かと思ったけど通った。pypyAC1.1秒。
　ループ内がシンプルなのと、区間だから実質半分なのが良かったのかな。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

H, W = MAP()
black = list2d(H+2, W+2, 0)
white = list2d(H+2, W+2, 0)
for i in range(1, H+1):
    row = LIST()
    for j in range(1, W+1):
        # 白黒を別グリッドに格納
        if (i+j)%2 == 0:
            black[i][j] = row[j-1]
        else:
            white[i][j] = row[j-1]

# 二次元累積和
for i in range(1, H+1):
    black[i] = list(accumulate(black[i]))
    white[i] = list(accumulate(white[i]))
for j in range(1, W+1):
    for i in range(1, H+1):
        black[i][j] += black[i-1][j]
        white[i][j] += white[i-1][j]

ans = 0
for i in range(1, H+1):
    for j in range(1, W+1):
        for k in range(i, H+1):
            for l in range(j, W+1):
                # 左上(i,j)と右下(k,l)の区間和を取って等しければ答え候補
                bsm = black[k][l] - black[k][j-1] - black[i-1][l] + black[i-1][j-1]
                wsm = white[k][l] - white[k][j-1] - white[i-1][l] + white[i-1][j-1]
                if bsm == wsm:
                    ans = max(ans, (k-i+1) * (l-j+1))
print(ans)
