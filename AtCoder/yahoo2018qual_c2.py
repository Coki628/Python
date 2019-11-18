# -*- coding: utf-8 -*-

"""
・蟻本演習3-4-2
・bitDP、ゲームDP、メモ化再帰
・自力はまだ全然無理だったけど、初めてやった時よりはお気持ちが分かった。
・bitDPパートはほぼ復習。ゲームDPパートはターンいらない形でまとめた。
・速度つらいのは変わらず。pypyでもTLE。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
X = LIST()
C = LIST()
V = LIST()

# dp[i][S] := 売った財宝がi個、残っている商品の状態が集合Sの時に、購入する場合の価値の最大値
dp = list2d(N+1, 1<<N, 0)
for i in range(1, N+1):
    # 財宝売却は前から順に行うのでN通りしかない
    yen = 0
    for j in range(i):
        yen += X[j]
    # 残商品の状態Sを全列挙
    for S in range(1<<N):
        c = v = 0
        for j in range(N):
            if S >> j & 1:
                c += C[j]
                v += V[j]
        # 今回の集合Sを全購入できるなら値を入れる
        if c <= yen:
            dp[i][S] = v
    # 各Sについて、全ての部分集合から貰うDPっぽく遷移させて更新
    for S in range(1<<N):
        for j in range(N):
            if S >> j & 1:
                dp[i][S] = max(dp[i][S], dp[i][S ^ (1<<j)])

memo = list2d(N+1, 1<<N, -1)
def rec(cur, S):
    if memo[cur][S] != -1:
        return memo[cur][S]
    # 財宝売却と相手側の商品販売停止
    res = 0
    if cur < N:
        res = INF
        for i in range(N):
            if S >> i & 1:
                # 相手側は、各商品販売停止から最小を取る
                res = min(res, rec(cur+1, S ^ (1<<i)))
    # 上記財宝売却と、この時点での商品購入で最大を取る
    res = max(res, dp[cur][S])

    memo[cur][S] = res
    return res

print(rec(0, (1<<N)-1))
