# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/tkppc3/editorial.pdf
・期待値系
・部分点解法(bitDP練習)
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

def bit_cnt(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

N=INT()

# dp[i] := 既に出た目を集合iとして、その状態になりうる通り数
dp=[0]*(1<<N)
for i in range(N):
    # 初期化：それぞれ1つだけ出した所に1通り
    dp[1<<i]=1

for bit in range(1, (1<<N)-1):
    if dp[bit]==0: continue
    for i in range(N):
        # iが既にbitに含まれている時はskip
        if bit>>i&1: continue
        # 集合bitから新たにi(に相当する位置にある目)を取った場合を遷移させる
        dp[bit+(1<<i)]+=dp[bit]

for bit in range(1, 1<<N):
    # 重複なく出た目の通り数*その次で重複する通り数*回数が少ない所の調整
    dp[bit]*=bit_cnt(bit)*N**(N-bit_cnt(bit))

sm=0
total=sum(dp)
for bit in range(1, 1<<N):
    # 集合bitで終わった時の獲得金額tmp
    tmp=0
    for i in range(N):
        if bit>>i&1:
            tmp+=i+1
    # 獲得金額*そうなる確率(通り数/全体)
    sm+=tmp*dp[bit]/total
# 全部足して全体の期待値
print(sm)
