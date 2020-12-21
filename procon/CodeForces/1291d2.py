# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/73563
　　　https://twitter.com/drken1215/status/1224009547787882496
　　　https://twitter.com/yamerarenaku/status/1224008460020277249
・アナグラム同士を分割してアナグラムにならないようにできるか
・長さ1はそもそも分割できないので例外的にYesとして、それ以外は
　種類数が3以上あればYesで2以下なら先頭末尾が違うならYes同じならNo。
・公式解説から、ようは先頭から異なる文字に変えていければOK、みたいは話なのは理解。
　でもこれ自分で導けるか？うーん。。
・種類数がいくつ含まれるかの確認は、文字毎26個の累積和を予め作っておけばOK。
"""

import sys
from itertools import accumulate

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

S = [ord(c)-97 for c in input()]
N = len(S)

acc = list2d(26, N+1, 0)
for i, c in enumerate(S):
    acc[c][i+1] += 1
for c in range(26):
    acc[c] = list(accumulate(acc[c]))

for _ in range(INT()):
    l, r = MAP()
    l -= 1
    ln = r - l
    if ln == 1:
        Yes()
        continue
    cnt = 0
    for c in range(26):
        if acc[c][r] - acc[c][l] >= 1:
            cnt += 1
    if cnt >= 3:
        Yes()
        continue
    if S[l] != S[r-1]:
        Yes()
    else:
        No()
