# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/agc035/editorial.pdf
・XOR系、Yesのケースは限定的なので、うまく条件を作る。
・自力では全然解法出ず。
・解説見れば納得できるけど、0と他の数xがあってYesになる条件とか、
　自分じゃ絶対探せない気がする…。
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
A=LIST()

if all([a==0 for a in A]):
    Yes()
    exit()

if N%3!=0:
    No()
    exit()

C=Counter(A)
if len(C.keys())==2:
    for k, v in C.items():
        if k==0:
            if v!=N//3:
                No()
                exit()
        else:
            if v!=N//3*2:
                No()
                exit()
    Yes()
elif len(C.keys())==3:
    xor=0
    for k, v in C.items():
        xor^=k
        if v!=N//3:
            No()
            exit()
    if xor==0:
        Yes()
    else:
        No()
else:
    No()
