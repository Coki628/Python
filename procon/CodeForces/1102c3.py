# -*- coding: utf-8 -*-

"""
ターン数が10^100なのに1000ターン終了だと思い込んでた…。
問題文ちゃんと読めてればすぐ解けたのになー。
"""

from bisect import bisect_right

N, X, Y = map(int, input().split())
aN = list(map(int, input().split()))
aN.sort()
weaks = bisect_right(aN, X)

# 同率or守りのが強い
if X <= Y:
    # 一撃で壊せるドアの半分切り上げ
    print(-(-weaks // 2))
# 攻めのが強い
else:
    print(len(aN))
