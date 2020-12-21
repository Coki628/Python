# -*- coding: utf-8 -*-

"""
参考：http://arc053.contest.atcoder.jp/data/arc/053/editorial.pdf
・回文いけるかと思いきや自力ならず。。
・式変形
"""

import sys
from collections import Counter

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

S = input()

C = Counter(S)
oddcnt = 0
for k, v in C.items():
    if v%2 == 1:
        oddcnt += 1

# 例外処理：奇数なし
if oddcnt == 0:
    print(len(S))
    exit()

# 各奇数文字列から1字ずつ引いて残り全部を偶数とする
evencnt = len(S) - oddcnt
# その偶数文字列のペア数
evenpair = evencnt // 2
# それらの偶数文字列を、先の取り出した1字奇数に均等に割り振って(切り捨てれば少ない方とみなせる)、
# *2でペアを戻して、+1で1字奇数の分を足す
print((evenpair//oddcnt)*2+1)
