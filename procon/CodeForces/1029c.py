# -*- coding: utf-8 -*-

"""
・終了後、さらに20分くらい粘って何とか自力AC。。
・座圧、いもす、座圧した分の重み付けをして累積和
・バグらせまくった。疲れた。いもすしたやつをさらにもう1回累積和とか。。最終的にちゃんとまとまったからいいけど。
・座圧した区間も、各位置の重みを取っておけばちゃんと区間長が復元できるのが分かったのは良かったかな。
・計算量は30万のNに左端右端があるので、実質60万に色々定数倍って感じ。pypyAC1.3秒。
"""

import sys
from itertools import accumulate

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def compress(S):
    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N = INT()
S = set()
LR = []
for _ in range(N):
    l, r = MAP()
    S.add(l)
    S.add(r)
    LR.append((l, r))

zipped, unzipped = compress(S)
M = len(zipped)
for i in range(N):
    l, r = LR[i]
    LR[i] = (zipped[l], zipped[r])

# 座圧した後の各値に重み付けする
W = [0] * M
for i in range(M-1):
    W[i] = unzipped[i+1] - unzipped[i]

# いもすでどこに何個の区間が重なっているかを取れるようにする
imos = [0] * M
for l, r in LR:
    imos[l] += 1
    imos[r] -= 1
imos = list(accumulate(imos))

# N個重なっている場所とN-1個重なっている場所を取れるようにする
acc1 = [0] * M
acc2 = [0] * M
for i in range(M):
    if imos[i] == N:
        acc1[i] = W[i]
    elif imos[i] == N-1:
        acc2[i] = W[i]
# 各所を重み付きの2値化みたいなことをして累積和
acc1 = [0] + list(accumulate(acc1))
acc2 = [0] + list(accumulate(acc2))

ans = 0
for l, r in LR:
    # 今回のLRが覆う中で元々N個重なっていた区間長(N-1になるので足す)
    seg1 = acc1[r] - acc1[l]
    # 今回のLRが覆う中で元々N-1個重なっていた区間長(N-1でなくなるので引く)
    seg2 = acc2[r] - acc2[l]
    # 元々N-1個重なっていた区間全体、から上記2つを足し引き
    ans = max(ans, acc2[-1] + seg1 - seg2)
print(ans)
