# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/dwacon2018-prelims/editorial.pdf
　　　https://www.hamayanhamayan.com/entry/2018/01/14/103600
・蟻本演習2-3-7、分割数
・もしcalcで分割数をかけ合わせる所がなければ、重複組み合わせを愚直DPした感じになる。
・同じkill数の人をグループにまとめることで、グループ内の通り数は分割数で出して、
　各グループの通り数は重複組み合わせのDPで数え上げる、と言った形にできる。
・計算量はN*max(sum(killA, killB))**2=100*1000*1000=1億、つらいかと思ったけど、
　pypyAC1.4秒、枝刈り入れたら1.0秒になった。
"""

import sys
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
INF = float('inf')
MOD = 10 ** 9 + 7

# 分割数：自然数NをK個の0以上の整数の和で表す場合の数
def partition(N, K):
    # dp[i][j] := jのi分割の総数
    dp = list2d(K+1, N+1, 0)
    dp[0][0] = 1
    for i in range(1, K+1):
        for j in range(N+1):
            dp[i][j] += dp[i-1][j]
            if j - i >= 0:
                dp[i][j] += dp[i][j-i]
            dp[i][j] %= MOD
    return dp

N, M = MAP()
killA = LIST()
killB = LIST()

kcntA = Counter(killA)
kcntB = Counter(killB)
ksmA = sum(killA)
ksmB = sum(killB)

P = partition(1000, 1000)
def calc(n, k):
    if n == 0:
        # デス0回(n=0)は分割数で処理できないので別処理
        return 1
    else:
        # n以下の自然数をk個のグループに割り振る(分割数)
        return P[k][n]

def solve(killA, killB, kcntA, ksmB):
    # dp[i][j] := i番目までのグループにj回デスを割り当てる時の通り数
    dp = list2d(len(kcntA)+1, ksmB+1, 0)
    dp[0][0] = 1
    for i, v in enumerate(kcntA.values()):
        for j in range(ksmB+1):
            for x in range(ksmB+1):
                # 枝刈り
                if j+x > ksmB:
                    break
                dp[i+1][j+x] += dp[i][j] * calc(x, v)
                dp[i+1][j+x] %= MOD
    return dp[-1][ksmB]

ansA = solve(killA, killB, kcntA, ksmB)
ansB = solve(killB, killA, kcntB, ksmA)
print(ansA*ansB%MOD)
