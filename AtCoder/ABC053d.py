# -*- coding: utf-8 -*-

"""
・速攻自力AC
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

from collections import Counter

N = int(input())
aN = list(map(int, input().split()))

# 重複分と最後の1つを分ける
dups = Counter(aN)
for k, v in dups.items():
    dups[k] -= 1
aN = sorted(set(aN))

# 重複分が偶数ならそっちだけでキレイに消せる
if sum(dups.values()) % 2 == 0:
    print(len(aN))
else:
    print(len(aN)-1)
