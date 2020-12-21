# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/agc031/editorial.pdf
　　　https://www.youtube.com/watch?v=g2LMI15ha_U
・200点ACならず…でもAGCのAはABCのD相当くらいだって動画でりんごさん言ってた。
・各文字独立で考える。この考察出なかったー…。
"""

import sys
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
S=input()

C=Counter(S)
ans=1
for v in C.values():
    # 各文字は1回しか使えないので、文字毎の場合の数は出現回数+1(使わない分)
    ans*=v+1
    ans%=MOD
# 最後に空文字列の分を引く
ans-=1
print(ans)
