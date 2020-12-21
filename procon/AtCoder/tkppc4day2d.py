# -*- coding: utf-8 -*-

"""
・400点自力ならず。
・式変形でxとy+zを一意にするまでは分かったんだけど、
　y+zのペアをうまく数える方法は分からなかった。
・後ろからやる。
・各ペアの片方の数をdictに持っておけば、逆が来た時にはそれを見て作れるペア数が分かる。
"""

import sys
from collections import defaultdict

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

N, P, Q = MAP()
A = LIST()

# 整数範囲で対応出来ない値なら例外処理
if (P+Q) % 2 or (P-Q) % 2:
    print(0)
    exit()

# 式変形でxとy+zは一意になる
x = (P+Q) // 2
yz = (P-Q) // 2
z_dict = defaultdict(int)
ans = yz_cnt = 0

for i in range(N-1, -1, -1):
    a = A[i]
    # xが来たら、ここまでのyzペアの数に応じて答えを追加
    if a == x:
        ans += yz_cnt
    # これまでに出現したyzペアの反対側の数に応じてyzペアの数を増やす
    yz_cnt += z_dict[yz-a]
    # 今回のaをyzペア候補の数に追加しておく
    z_dict[a] += 1

print(ans)
