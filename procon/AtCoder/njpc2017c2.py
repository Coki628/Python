# -*- coding: utf-8 -*-

"""
・500点自力AC！
・メモ化再帰
・考察とか時間かかりすぎだけどな…。
"""

import sys
from bisect import bisect_left, bisect_right

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

N,L=MAP()
# 小数を考慮するため、2倍して考えておく
L*=2
A=[-1]
A+=[INT()*2 for i in range(N)]
N+=1
Aset=set(A)
# [次に跳ぶハードル] = 前のハードルからの距離
memo=[INF]*(N+1)

def rec(cur, h):
    # 最後のハードルより先まで行けたら成功
    if h==N:
        YES()
        exit()
    # 一度見たところはメモより条件が良くなければやらない
    if memo[h]<=cur:
        return
    memo[h]=cur
    # ギリギリまで走って次のハードルを跳んだ時の位置
    mx_dist=A[h]-1+L
    idx=bisect_left(A, mx_dist)-1
    # 届く範囲で遠くにあるやつから順にギリギリに跳び越えてみる
    for i in range(idx, -1, -1):
        if A[i]<=A[h-1]+cur:
            break
        # 跳ぶ位置jmpを決める
        if h==1:
            jmp=max(A[h-1]+cur, A[i]+1-L)
        else:
            # 初回以外は跳べない区間Lを考慮する
            jmp=max(A[h-1]+cur+L, A[i]+1-L)
        if jmp+L not in Aset:
            a=bisect_left(A, jmp+L)
            b=bisect_right(A, jmp+L+L)
            # 次の跳べない区間にハードルが無いようならいける
            if a==b:
                rec(jmp+L-A[i], i+1)

rec(1, 1)
NO()
