# -*- coding: utf-8 -*-

"""
・今日の一問から
・400点自力ACだけど、まあ総当たりしただけみたいで微妙。。
　想定解じゃなかったし、嘘解法じゃないといいんだけど。。
"""

import sys

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

H, W = MAP()

if H%3==0 or W%3==0:
    print(0)
    exit()

def calc(h1, h2, W):
    # 3つの領域の面積abc
    a = h1 * W
    b = h2 * (W//2)
    c = h2 * ceil(W, 2)
    return max(abs(a-b), abs(b-c), abs(c-a))

ans = INF
# 2つと1つに区切るなら、/3切り捨てか切り上げが最適のはず
h1 = H // 3
h2 = H - h1
ans = min(ans, calc(h1, h2, W))
h1 = ceil(H, 3)
h2 = H - h1
ans = min(ans, calc(h1, h2, W))

# 縦横逆もやる
H, W = W, H
h1 = H // 3
h2 = H - h1
ans = min(ans, calc(h1, h2, W))
h1 = ceil(H, 3)
h2 = H - h1
ans = min(ans, calc(h1, h2, W))

# 3つ1方向に並べるパターン
ans = min(ans, abs(ceil(W, 3)*H-W//3*H))
ans = min(ans, abs(ceil(H, 3)*W-H//3*W))

print(ans)
