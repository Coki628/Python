# -*- coding: utf-8 -*-

"""
・蟻本演習3-4-1、前すごい悩んだのは覚えてたのに自力ならず。。。
・bitDP、累積和
・前書いたコードさらっと見て概要掴んだらすぐ組めたけど。。
・累積和を各ぬいぐるみ毎に作っておくとか、ビットからその時の開始位置を数えたりとか、
　色々モノにしたい考察があるよね。。。
・前とほぼ同じコードになったのに、なぜか速度改善しててpypyAC1.7→1.2秒。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, M = MAP()
A = [a-1 for a in LIST(N)]
C = Counter(A)

# 初期状態で各ぬいぐるみが区間に何個あるのかをO(1)で取れるように準備する
acc = list2d(M, N+1, 0)
for i, a in enumerate(A):
    acc[a][i+1] += 1
for i in range(M):
    acc[i] = list(accumulate(acc[i]))

# dp[i] := 位置を決めたぬいぐるみの種類を集合iとして、そこまでに動かさなくていい数の最大値
dp = [0] * (1<<M)
# 初期値：一番左に各ぬいぐるみを並べた時の値
for i in range(M):
    dp[1<<i] = acc[i][C[i]]

for bit in range(1<<M):
    # ぬいぐるみiが集合bitに含まれていたら、その数を開始位置に足す
    cur = 0
    for i in range(M):
        if bit >> i & 1:
            cur += C[i]
    # 各ぬいぐるみiについて、今回の位置にを置く遷移を行う
    for i in range(M):
        if bit >> i & 1:
            continue
        # ぬいぐるみiを収める予定の区間にiがある個数を取る
        cnt = acc[i][cur+C[i]] - acc[i][cur]
        dp[bit|1<<i] = max(dp[bit|1<<i], dp[bit]+cnt)
# 全体 - 動かさなくていい数
print(N-dp[-1])
