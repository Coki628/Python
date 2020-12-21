# -*- coding: utf-8 -*-

"""
・D自力AC
・桁DPっぽく。桁DPやったばっかりだったのもあってうまくいった。
"""

import sys

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

N=input()
lenN=len(N)

# 例外処理
if N=='1':
    print(1)
    exit()

# dp[i] := 大きい方からi+1桁目が1になる回数
dp=[0]*lenN
if int(N[0])==1:
    dp[0]=int(N[1:])+1
else:
    dp[0]=10**(lenN-1)

memo=0
for i in range(1, lenN):
    # 前の桁に影響される回数(この値は次の桁のために取っておく)
    memo+=int(N[i-1])*10**(lenN-1-i)
    dp[i]+=memo
    # 今の桁が1
    if int(N[i])==1:
        # 自分より小さい全桁の値+1
        dp[i]+=1
        if i!=lenN-1:
            dp[i]+=int(N[i+1:])
    elif int(N[i])>=2:
        # 2以上ならこの桁の1は全て登場済
        dp[i]+=10**(lenN-1-i)

# 合計すれば全体での1の登場回数
print(sum(dp))
