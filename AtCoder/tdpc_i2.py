# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-10
・区間DP
・内容理解してから、自分で作りかけて出来なかったこっちのメモ化再帰も完成させて通した。
・pythonTLE、pypyAC0.3秒。普通のDPより遅くなるかと思ったら意外とちょこっとだけ速くなってた。
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

S = input()
N = len(S)

memo = list2d(N+1, N+1, -1)
def rec(l, r):
    if memo[l][r] != -1: 
        return memo[l][r]

    # 区間が3以下の場合はすぐ結果を返す
    if r-l == 3:
        if S[l:r] == 'iwi':
            memo[l][r] = 1
            return 1
        else:
            memo[l][r] = 0
            return 0
    elif r-l < 3:
        memo[l][r] = 0
        return 0
    
    # [l,m)と[m,r)の全てから最適な値を集めてくる
    res = -INF
    for m in range(l+1, r):
        res = max(res, rec(l, m)+rec(m, r))
    
    # 連鎖の可能性がある場合
    if (r-l) % 3 == 0 and S[l] == S[r-1] == 'i':
        # i***(全消し)***wiが成立するなら+1
        if (r-l)//3-1 == rec(l+1, r-2) and S[r-2] == 'w':
            res = max(res, rec(l+1, r-2)+1)
        # iw***(全消し)***iが成立するなら+1
        if (r-l)//3-1 == rec(l+2, r-1) and S[l+1] == 'w':
            res = max(res, rec(l+2, r-1)+1)

    memo[l][r] = res
    return res

print(rec(0, N))
