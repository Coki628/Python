# -*- coding: utf-8 -*-

"""
・時間かかったけど500点自力AC！
・とはいえ実戦なら時間切れだっただろうけど。。
・最強候補を見つけてそいつで実際に全部と試す、っていう前問の方針でいけた。
・でも想定解じゃなかったから、そっちも後でやってみよう。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from functools import reduce

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

def lcm(x, y): return (x * y) // gcd(x, y)
def gcd_list(li): return reduce(gcd, li, 0)

N=INT()

A=[]
AB=[]
for i in range(N):
    a,b=MAP()
    A.append(lcm(a, b))
    AB.append((a, b))

# 一番うまくいきそうな(共通する場所が多そうな)数を答え候補とする
ans=gcd_list(A)
for a, b in AB:
    # その答え候補で実際にaかb最適な方を選んでいく
    ans=max(gcd(ans, a), gcd(ans, b))
print(ans)
