# -*- coding: utf-8 -*-

"""
参考：http://arc058.contest.atcoder.jp/data/arc/058/editorial.pdf
・蟻本演習3-4-3
・bitDP、スライドbitDP
・自力ACは無理だったけど、公式解説の説明だけで理解して正しく実装できたからまあ良しとする。
・1~10までの値がある直前のいくつかの情報をどうビットに落とし込んでいいか分からなかった。
・やり方としては、要素が3個あるから3ビット使って、みたいにではなく、
　今回の値が3なら3ビットずらして3ビット目に詰めるし、みたいな感じ。
　取っておく個数は関係なくて、直前の値の合計がXYZまで分かればいいから、これでうまくいく。
・あと、やりやすさのために余事象でやる。このあたりも考察で気付けるようになりたい。
・計算量は40*2^16*10=約2600万で、pythonTLE(制約4秒)、pypyAC1.0秒。
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

N, X, Y, Z = MAP()
M = X + Y + Z

# XYZとの一致をチェックするビット列
check = 1<<(X+Y+Z-1) | 1<<(Y+Z-1) | 1<<(Z-1)
dp = list2d(N+1, 1<<(M-1), 0)
# dp[i][S] := i個目まで見て、前の値M-1以下までの集合がSの時の、XYZを含まないものの通り数
dp[0][0] = 1
msk1 = (1<<M) - 1
msk2 = (1<<(M-1)) - 1
for i in range(N):
    for S in range(1<<(M-1)):
        if not dp[i][S]:
            continue
        for j in range(10):
            # 1ビットずらして1を足してjビットずらしてマスクする
            S2 = (S<<1|1)<<j & msk1
            # 直前Mビットで使った値がXYZと一致していないか確認する
            if (S2 & check) != check:
                dp[i+1][S2&msk2] += dp[i][S]
                dp[i+1][S2&msk2] %= MOD
total = pow(10, N, MOD)
print((total-sum(dp[N])%MOD)%MOD)
