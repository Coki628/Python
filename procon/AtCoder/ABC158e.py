# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc158/editorial.pdf
　　　https://www.youtube.com/watch?v=sv3l8heocMY
・500点自力ならず。。P=2,5の例外処理が気付けなかったー。
・zero sum ranges系、後ろから累積和、例外処理
・modを取りながら、後ろから累積和っぽいものを作っておけば、
　区間和が0になる場所が分かるのでzero sum ranges系の典型になる。(ここまでは一応出来てた。)
・ただし、この区間和取得のためには、左右を足し引きした数を10^kで割る必要が出てくるので、
　modが10で割り切れる数(2,5)だと0除算になってダメになる。よって別処理する。(これが分からなかった。。)
"""

import sys
from itertools import accumulate
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

N, P = MAP()
A = list(map(int, input()))

# mod2,5は例外処理
if P == 2 or P == 5:
    ans = 0
    for i, a in enumerate(A):
        if a % P == 0:
            ans += i + 1
    print(ans)
    exit()

# 後ろから累積和っぽくする
acc = [0] * (N+1)
cur = 0
k = 0
for i in range(N-1, -1, -1):
    cur += pow(10, k, P) * A[i]
    cur %= P
    acc[i] = cur
    k += 1

# 同値ペアの組み合わせ数を数える
C = Counter(acc)
ans = 0
for v in C.values():
    ans += v * (v-1) // 2
print(ans)
