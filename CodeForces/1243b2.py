# -*- coding: utf-8 -*-

"""
・終了後自力AC
・問題のスワップのルール勘違い(自分側同士でも交換できる気でいた)しててタイムロスした。
・最初から正しい方針でやれてれば時間内いけたぽいからちょっと残念。。
"""

import sys
from collections import Counter

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def check1(C):
    for v in C.values():
        # 各文字が半分ずつないとそもそも無理
        if v % 2 != 0:
            return False
    return True

def check2(S, T):
    ans = []
    # 前から1文字ずつ確認
    for i in range(N):
        # 一致しない場所を見つけたら操作する
        if S[i] != T[i]:
            # 欲しい文字が自分側にあるなら、1回スワップでOK
            try:
                j = S[i+1:].index(S[i]) + i + 1
                S[j], T[i] = T[i], S[j]
                ans.append((j, i))
            # なければ、1回自分側に移動させてからもう1回スワップする
            except:
                j = T[i+1:].index(S[i]) + i + 1
                S[i], T[j] = T[j], S[j]
                ans.append((j, j))
                S[j], T[i] = T[i], S[j]
                ans.append((j, i))
    return ans

for _ in range(INT()):
    N = INT()
    S = list(input())
    T = list(input())
    C = Counter(S+T)
    if not check1(C):
        No()
        continue
    ans = check2(S, T)
    Yes()
    print(len(ans))
    for i, j in ans:
        print(i+1, j+1)
