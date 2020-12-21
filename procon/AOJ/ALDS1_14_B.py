# -*- coding: utf-8 -*-

"""
参考：http://perogram.hateblo.jp/entry/2019/05/21/235639
・ローリングハッシュ
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
INF = float('inf')
MOD = 10 ** 9 + 7

T = input()
P = input()

# 文字列Sの中で文字列Tを含む場所のindexを返却
def rolling_hash_search(S, T):

    ans = []
    slen = len(S)
    tlen = len(T)
    # 例外処理
    if slen < tlen:
        return ans
    MOD = 2 ** 64
    b = 10 ** 8 + 7
    a = 1
    for i in range(tlen):
        a *= b
        a %= MOD
    
    # 先頭のローリングハッシュ
    s_hash = 0
    for i in range(tlen):
        s_hash = s_hash * b + ord(S[i])
        s_hash %= MOD
    t_hash = 0
    for i in range(tlen):
        t_hash = t_hash * b + ord(T[i])
        t_hash %= MOD
    
    # Sの始点を1つずつ進めながらハッシュ値をチェック
    for i in range(slen-tlen+1):
        if s_hash == t_hash:
            ans.append(i)
        if i+tlen < slen:
            s_hash = s_hash * b - ord(S[i]) * a + ord(S[tlen+i])
            s_hash %= MOD
    return ans

res = rolling_hash_search(T, P)
if len(res):
    [print(i) for i in res]
