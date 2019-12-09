# -*- coding: utf-8 -*-

"""
・基本方針としては問題なかった。
・公式解は数え上げる時にぶたんじゃないけど、それはこっちでも全然OK。
・直したのは重複調整の部分だけ。該当マスだけ確認すればいいっていう所に至れなくて残念。。
"""

import sys
from bisect import bisect_left, bisect_right

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

R, C, K = MAP()
N = INT()
RCNT = [0] * R
CCNT = [0] * C
RC = []
for i in range(N):
    r, c = MAP()
    r -= 1; c -= 1
    RCNT[r] += 1
    CCNT[c] += 1
    RC.append((r, c))

CSORT = sorted(CCNT)
ans = 0
for r in range(R):
    rcnt = RCNT[r]
    # c+rがKになる
    idx1 = bisect_left(CSORT, K-rcnt)
    idx2 = bisect_right(CSORT, K-rcnt)
    ans += idx2 - idx1

# 重複調整
for r, c in RC:
    # このマス合わせてのKだと重複のせい(本当は足りない)なので引く
    if RCNT[r] + CCNT[c] == K:
        ans -= 1
    # このマス合わせてのK+1は、本当はKなので足す
    if RCNT[r] + CCNT[c] == K + 1:
        ans += 1
print(ans)
