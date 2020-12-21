# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc073/editorial.pdf
　　　https://ei1333.hateblo.jp/entry/2017/05/01/004235
　　　https://atcoder.jp/contests/abc060/submissions/3782609
・公式解
・累積和あり
・速度は累積和なしの半分強ってとこ。
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

from collections import defaultdict

N, W = map(int, input().split())
wN = [0] * N
vN = [[] for i in range(4)]
for i in range(N):
    w, v = map(int, input().split())
    if i == 0:
        offset = w
    vN[w-offset].append(v)

# 重さ毎のリストを降順ソート
for i in range(4):
    # 空リストのままだとこの後ループを回ってくれないので、[0]を追加する
    vN[i] = [0] + sorted(vN[i], reverse=True)

# 累積和
for i in range(4):
    for j in range(1, len(vN[i])):
        vN[i][j] = vN[i][j-1] + vN[i][j]

ans = 0
for i in range(len(vN[0])):
    for j in range(len(vN[1])):
        for k in range(len(vN[2])):
            for l in range(len(vN[3])):
                # 重さリミットを超えるパターンは対象外
                if offset * i + (offset+1) * j + (offset+2) * k + (offset+3) * l > W:
                    continue
                ans = max(vN[0][i] + vN[1][j] + vN[2][k] + vN[3][l], ans)
print(ans)
