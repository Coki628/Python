# -*- coding: utf-8 -*-

"""
・600点無理そうだし、割り切って300点狙いの低速実装
・ってこれ300点ケースも間に合わないし…。
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

from bisect import bisect_right

N, K = map(int, input().split())
aN = list(map(int, input().split()))

aN.sort()
# 単体で条件を満たすものは必要確定なので除く
aN2 = aN[:bisect_right(aN, K)]
aN2.reverse()

need = [False] * len(aN2)
# 大きい方から試したいので逆順でループ回す
for i in range((1 << len(aN2))-1, -1, -1):
    # 全部合わせても足りないようじゃ論外
    if sum(aN2) < K:
        break
    sm = 0
    for j in range(len(aN2)):
        # ビット全探索で部分集合を作る
        if 1 & i>>j == 1:
            sm += aN2[j]
        # 複数で条件を満たせば、それに含まれるものは必要
        if sm >= K:
            for k in range(j+1):
                if 1 & i>>k == 1:
                    need[k] = True
            break
    # 最後まで足りなかったら、これ以降も足りないので終了
    else:
        break
print(need.count(False))


