# -*- coding: utf-8 -*-

"""
・600点自力AC！やったね。
・あーやっぱ貪欲無理かぁとか諦めかけたけど、実際はしょうもないとこ
　(ループの終了位置)でWAしてて、ちゃんと粘ってバグ見つけて良かった。
"""

import sys
from collections import Counter

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

C = Counter(A)
ans = 0
# 大きい数から見ていく
for i in range(30, 0, -1):
    x = 2 ** i
    for k in C.keys():
        if x-k in C:
            # 同値ペアは別処理
            if k == x-k:
                p = C[k] // 2
                C[k] -= p * 2
                ans += p
            else:
                # 少ない方に合わせてペア作成
                p = min(C[k], C[x-k])
                C[k] -= p
                C[x-k] -= p
                ans += p
print(ans)
