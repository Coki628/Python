# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-8
・3次元座標圧縮、3次元いもす
・試行錯誤の末、うまくいった。
・愚直と比べて、pypy1.0秒→0.3秒AC、pythonTLE→1.6秒ACと高速化。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def compress(A):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(set(A))):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N, K = MAP()
X = []
Y = []
Z = []
XYZ = []
for i in range(N):
    x1, y1, z1, x2, y2, z2 = MAP()
    X += [x1, x2]
    Y += [y1, y2]
    Z += [z1, z2]
    XYZ.append((x1, y1, z1, x2, y2, z2))
 
# 座標圧縮
X_zipped, X_unzipped = compress(X)
Y_zipped, Y_unzipped = compress(Y)
Z_zipped, Z_unzipped = compress(Z)
 
for i in range(N):
    # 圧縮後の値に置き換え
    x1, y1, z1, x2, y2, z2 = XYZ[i]
    XYZ[i] = (X_zipped[x1], Y_zipped[y1], Z_zipped[z1], \
              X_zipped[x2], Y_zipped[y2], Z_zipped[z2])
 
# ここから圧縮後の値でやる
H, W, D = len(Y_zipped), len(X_zipped), len(Z_zipped)
# 3次元いもす
imos = list3d(H, W, D, 0)
for i in range(N):
    x1, y1, z1, x2, y2, z2 = XYZ[i]
    imos[y1][x1][z1] += 1
    imos[y2][x1][z1] -= 1
    imos[y1][x2][z1] -= 1
    imos[y1][x1][z2] -= 1
    imos[y2][x2][z1] += 1
    imos[y1][x2][z2] += 1
    imos[y2][x1][z2] += 1
    imos[y2][x2][z2] -= 1
# 3方向から累積和
for i in range(H):
    for j in range(W):
        for k in range(D-1):
            imos[i][j][k+1] += imos[i][j][k]
for k in range(D):
    for i in range(H):
        for j in range(W-1):
            imos[i][j+1][k] += imos[i][j][k]
for j in range(W):
    for k in range(D):
        for i in range(H-1):
            imos[i+1][j][k] += imos[i][j][k]
 
ans = 0
for i in range(H-1):
    for j in range(W-1):
        for k in range(D-1):
            if imos[i][j][k] >= K:
                # 元の値に戻して体積を出す
                x = X_unzipped[j+1] - X_unzipped[j]
                y = Y_unzipped[i+1] - Y_unzipped[i]
                z = Z_unzipped[k+1] - Z_unzipped[k]
                ans += x * y * z
print(ans)
