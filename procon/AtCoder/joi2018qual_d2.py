# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2018/2019-yo/2019-yo-t4-review.html
　　　https://atcoder.jp/contests/joi2019yo/submissions/6630347
・実は自分の左右だけ見れば、沈んだ時の増減が分かる。
　って言う点に気付けなかった。。
・解説だけで自力実装うまくいかなくて人の参考。
　その区画が沈んだかどうかのフラグを持つ。
　沈めて確認、を1つずつやっていけば、連続した区間は1回しか数えないからうまくいく。
"""

import sys
from collections import defaultdict

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

N = INT()
A = LIST()

# 最初から陸地0
if max(A) == 0:
    print(0)
    exit()

# その区画がまだ陸地かどうかのフラグ
is_land = [False] + [True]*N + [False]

B = defaultdict(list)
for i, a in enumerate(A):
    # 高さ毎に区画の位置を格納
    B[a].append(i+1)

cnt = mx = 1
# 高さaのループ
for a, b in sorted(B.items()):
    # その高さでの各区画iのループ
    for i in b:
        # 区画iが沈む
        is_land[i] = False
        # 両側が陸地なら島が増える
        if is_land[i-1] and is_land[i+1]:
            cnt += 1
        # 両側が海なら島が減る
        elif not is_land[i-1] and not is_land[i+1]:
            cnt -= 1
    # 各高さでの島の数からmaxを取る
    mx = max(mx, cnt)
print(mx)
