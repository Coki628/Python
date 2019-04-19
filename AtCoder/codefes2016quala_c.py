# -*- coding: utf-8 -*-

"""
・400点自力AC！
・文字列系、辞書順最小
・貪欲で攻めたらちゃんと通ったから良かった。
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

S=list(input())
N=len(S)
K=INT()

# とりあえず前から順に足りるところは全部aまで動かす
for i in range(N):
    toA=(ord('z')+1-ord(S[i]))%26
    if K>=toA:
        S[i]='a'
        K-=toA

# 残った分は最後の文字をあるったけ動かす
K%=26
S[-1]=chr((ord(S[-1])-97+K)%26+97)
print(''.join(S))
