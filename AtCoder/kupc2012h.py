# -*- coding: utf-8 -*-

"""
参考：http://www.kupc.jp/static/media/H.3dbd0fae.pdf
・蟻本演習3-2-3
・反転(ライツアウト)
・制約の、行列が偶数限定というのもポイント(全然見てなかった。。)
・該当マスがかかる行列全部やるとそのマスだけ反転するという考察。
・上記のように1つのマスだけを反転させる方法があったので、これで任意のマスを自由に操作できる。
・後は実装だるいけど4方向全部から累積和。
・計算量は1000*1000グリッドで4回累積和と最後の集計で、100万*5=500万。
・pythonTLEでpypyAC0.8秒。これ昔のでメモリ制限128Mなんだけど、pypyかなりMLE危ない116Mだった。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

H, W = MAP()
grid = [None] * H
for i in range(H):
    grid[i] = LIST()

# 4方向から累積和(グリッドで0のところで加算)
acc1 = list2d(H, W, 0)
for i in range(H):
    acc1[i][0] = grid[i][0] ^ 1
    for j in range(1, W):
        acc1[i][j] = acc1[i][j-1] + (grid[i][j] ^ 1)
acc2 = list2d(H, W, 0)
for i in range(H):
    acc2[i][-1] = grid[i][-1] ^ 1
    for j in range(W-2, -1, -1):
        acc2[i][j] = acc2[i][j+1] + (grid[i][j] ^ 1)
acc3 = list2d(H, W, 0)
for j in range(W):
    acc3[0][j] = grid[0][j] ^ 1
    for i in range(1, H):
        acc3[i][j] = acc3[i-1][j] + (grid[i][j] ^ 1)
acc4 = list2d(H, W, 0)
for j in range(W):
    acc4[-1][j] = grid[-1][j] ^ 1
    for i in range(H-2, -1, -1):
        acc4[i][j] = acc4[i+1][j] + (grid[i][j] ^ 1)

ans = list2d(H, W, 0)
for i in range(H):
    for j in range(W):
        # 全方向合計
        ans[i][j] = acc1[i][j] + acc2[i][j] + acc3[i][j] + acc4[i][j]
        # グリッドで0の所は4方向分重複するので-3
        if grid[i][j] == 0:
            ans[i][j] -= 3
        # 奇数回の場所は実施
        ans[i][j] %= 2
    print(*ans[i])
