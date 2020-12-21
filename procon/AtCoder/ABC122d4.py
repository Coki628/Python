# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc122/editorial.pdf
・数え上げDP
・NGの確認方法を公式解のやり方にしてみる。
"""

import sys
from itertools import product
from collections import Counter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

N=INT()

# dp[i][j] := 全部でi文字の時、最後の3文字がjとなる文字列の数
dp=[Counter() for i in range(N+1)]
keys=[''.join(key) for key in product('ACGT', repeat=3)]
# 3文字での初期値
for key in keys:
    if key not in ['AGC', 'GAC', 'ACG']:
        dp[3][key]=1

keys=[''.join(key) for key in product('ACGT', repeat=4)]
for i in range(4, N+1):
    for key in keys:
        for j in range(4):
            tmp=list(key)
            if j>=1:
                # 隣同士をスワップさせる
                tmp[j],tmp[j-1]=tmp[j-1],tmp[j]
            if ''.join(tmp).count('AGC'):
                break
        else:
            # 4パターンいずれもAGCを含まないものだけ遷移させる
            cur,nxt=key[:3],key[1:]
            dp[i][nxt]=(dp[i][nxt]+dp[i-1][cur])%MOD
print((sum(dp[N].values()))%MOD)
