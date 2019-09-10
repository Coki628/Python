# -*- coding: utf-8 -*-

"""
・自力AC
・制約がN=100だったから、多分この貪欲O(N)は通らんだろうと思ったら通った。
　一応やってみるもんだ。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

N, R = MAP()
S = input()

found = paint = False
lencnt = ans = 0
# 後ろから見る
for i in range(N-1, -1, -1):
    if S[i] == '.':
        found = True
    if found:
        lencnt += 1
        # 塗られてない範囲がRに達したら塗る
        if lencnt == R:
            ans += 1
            lencnt = 0
            # 塗ったことがあるフラグを立てる
            if not paint:
                paint = True
            found = False
    # 塗ったことがある状態(1マス目は除く)は移動コストを足す
    if paint and i != 0:
        ans += 1
else:
    # 最初まで来て、塗る場所が残っている状態なら塗る
    if found:
        ans += 1
print(ans)
