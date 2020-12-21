# -*- coding: utf-8 -*-

"""
参考：https://tenka1-2016-qualb.contest.atcoder.jp/data/other/tenka1-2016-qualb/editorial.pdf
　　　https://ei1333.hateblo.jp/entry/2016/08/28/112119
・結構みんな普通のDPしてたけど、遷移大変そうだったからメモ化再帰にした。
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

S=input()
N=len(S)

# memo[現在位置][ここまでの左括弧と右括弧の差]][最後の操作位置]
memo=list3d(N+1, N+1, N+1, -1)
def rec(cur, gap, last):

    # 右のが多くなったらNG
    if gap<0:
        return INF
    # メモ返却
    if memo[cur][gap][last]!=-1:
        return memo[cur][gap][last]
    # 終了位置
    if cur==N:
        # 左右が一致してればOK
        if gap==0:
            return last
        else:
            return INF

    res=INF
    # 左が来た時
    if S[cur]=='(':
        # 右に変える
        res=min(res, rec(cur+1, gap-1, cur)+1)
        # そのまま
        res=min(res, rec(cur+1, gap+1, last))
    # 右が来た時
    else:
        # 左に変える
        res=min(res, rec(cur+1, gap+1, cur)+1)
        # そのまま
        res=min(res, rec(cur+1, gap-1, last))

    # メモして返却
    memo[cur][gap][last]=res
    return res

print(rec(0, 0, 0))
