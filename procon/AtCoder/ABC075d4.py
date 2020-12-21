# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・試行錯誤の末、なんとか自力AC。。(C++でAC確認、これはpypyTLE)
・前の見たけど、頂点の全組*4回(重複あり)じゃなくて、
　x座標の全組2回(重複なし)*y座標の全組2回(重複なし)なら通ってた。。
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
# 20億^2で10^18を超えるので19乗で
INF = 10 ** 19
MOD = 10 ** 9 + 7

N, K = MAP()
XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))

ans = INF
# 4点組み合わせ全部試す
for i in range(N):
    x1, y1 = XY[i]
    for j in range(i, N):
        x2, y2 = XY[j]
        for k in range(j, N):
            x3, y3 = XY[k]
            for l in range(k, N):
                x4, y4 = XY[l]
                xl, xr = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
                yl, yr = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
                # 4点の内側にいくつ点があるか数える
                cnt = 0
                for x5, y5 in XY:
                    if xl <= x5 <= xr and yl <= y5 <= yr:
                        cnt += 1
                if cnt >= K:
                    ans = min(ans, (xr-xl) * (yr-yl))
print(ans)
