# -*- coding: utf-8 -*-

"""
・粘りに粘って自力AC！これがゆきこの茶diffか。。
・MODの扱い、偶奇
・結局、答えにはa, b, xは偶奇しか関係しない。それが分かれば場合分け。
・その考察に辿り着くのも時間かかったけど、その後もNの回数勘違いしてて
　(両方1回ずつで1回じゃなくて片方で1回かと思ってた)、そのことを発覚させるのに
　TLE想定の愚直まで書いてデバッグして、やっとのAC。。
・あと、問題文前半の小難しい前置きにだいぶ惑わされた感もある。。
"""

import sys

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

a, b, x, N = MAP()

# mod = 2**32
# def test(x):
#     acnt = bcnt = 0
#     abl = awh = bbl = bwh = 0
#     for i in range(N*2):
#         x = (a*x + b) % mod
#         if i % 2 == 0:
#             acnt += x % 6 + 1
#             if acnt % 2 == 1:
#                 abl += 1
#             else:
#                 awh += 1
#         else:
#             bcnt += x % 6 + 1
#             if bcnt % 2 == 1:
#                 bbl += 1
#             else:
#                 bwh += 1
#     aans = min(abl, awh)
#     bans = min(bbl, bwh)
#     return aans, bans
# print(*test(x))

a %= 2
b %= 2
x %= 2
# x1から奇、偶、奇、偶ってなる
if a == 1 and b == 1 and x == 0:
    ans1 = 0
    ans2 = N // 2
# x1から偶、奇、偶、奇ってなる
elif a == 1 and b == 1 and x == 1:
    ans1 = N // 2
    ans2 = 0
# 制約上、本当はこの先の場合分けいらない(やってしまったから残した)
elif a == 1 and b == 0 and x == 0:
    ans1 = N // 2
    ans2 = N // 2
else:
    ans1 = ans2 = 0
print(ans1, ans2)
