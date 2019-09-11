# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc028
・公式解、優先度付きキュー
・それはちょっと考えたけど、取得位置がK番目で先頭とも末尾とも限らないから使えないかなぁ
　って思って使わなかった。
・常にキューに保持するのをK人に保つことで、毎回末尾を取ればK番目を取得できることになる。
・やっぱりこっちのがBITにぶたんより速い(logいっこ外れたしね)。python0.2秒AC。
"""

import sys
from heapq import heappush, heappop

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

N, K = MAP()
X = [x-1 for x in LIST()]
xtoi = [0] * N
for i, x in enumerate(X):
    xtoi[x] = i

# Kまでは予め入れておく
que = []
for i in range(K):
    heappush(que, -X[i])

# 最初の1人は比較しなくていいので別処理
x = -heappop(que)
print(xtoi[x]+1)
heappush(que, -x)

for i in range(K, N):
    cur = X[i]
    prev = -heappop(que)
    # 今回のxと現在のK人の末尾を比較
    if cur < prev:
        # 今回の方が小さければ、それを新たにK人に含めて(古い末尾は捨てて)、再度末尾を取得
        heappush(que, -cur)
        prev = -heappop(que)
    # 今回の方が大きければ、今回の方を捨てた形になる
    print(xtoi[prev]+1)
    heappush(que, -prev)
