# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc141/editorial.pdf
　　　http://perogram.hateblo.jp/entry/2019/05/21/235639
・ローリングハッシュ
・でもこれはTLE。しかも文字列で普通にやったやつより遅い。。
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

N = INT()
S = input()

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """

    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

# ローリングハッシュで一致する文字列があるか確認
def rolling_hash_search(S, T):

    ans = []
    slen = len(S)
    tlen = len(T)
    # 例外処理
    if slen < tlen:
        return False
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
            return True
        if i+tlen < slen:
            s_hash = s_hash * b - ord(S[i]) * a + ord(S[tlen+i])
            s_hash %= MOD
    return False

def calc(m):
    for i in range(N-m*2+1):
        S1 = S[i:i+m]
        S2 = S[i+m:]
        # 文字列S2に文字列S1が含まれているか
        if rolling_hash_search(S2, S1):
            return True
    return False

print(bisearch_max(0, N, calc))
