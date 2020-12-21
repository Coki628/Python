# -*- coding: utf-8 -*-

"""
参考：http://kmjp.hatenablog.jp/entry/2012/09/15/0930
・自力ならず。ほぼ合ってる方針でやってたんだけどね。。
・(メモしない)再帰探索、フィボナッチ
・なるべく大きいやつから減らす、が最善か自信なかったからWAして諦めて解法見ちゃったんだけど、
　間違ってたのは1回で終わるケースを正しく処理できてなかったから。
・でも大きい方から貪欲OKなの非自明だよな。真ん中らへん2,3個使った方がうまくいくケースとか
　ありそうなのに。
"""

import sys
from bisect import bisect_left

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

# 和に使う可能性のある数を列挙
dp = [0] * 2
dp[1] = 1
i = 2
while dp[i-1] + dp[i-2] <= N:
    dp.append(dp[i-1] + dp[i-2])
    i += 1
M = len(dp)

def rec(cur):
    idx = bisect_left(dp, cur)
    # ぴったりの数があれば1回で済む
    if idx < M and dp[idx] == cur:
        return 1
    else:
        # なければなるべく大きいやつで減らして再帰的に調べる
        return rec(cur - dp[idx-1]) + 1
print(rec(N))
