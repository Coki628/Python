# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc027
・満点解法
・矢印の向きは自分より後(右)に出てくる+-にしか影響を与えない。
　その数さえ数えておけば、各矢印に見込める値を把握できる。
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

M=S.count('M')
# i+1回目の移動を右にした時に見込める値(左移動はこれの符号逆でOK)
A=[0]*M
mi=M-1
sm=0
# 自分より右に出現する+-を数える
for i in range(N-1, -1, -1):
    if S[i]=='M':
        A[mi]=sm
        mi-=1
    elif S[i]=='+':
        sm+=1
    else:
        sm-=1

# ソートして、大きい半分と小さい半分をそれぞれ正負に当て込めばOK
A.sort(reverse=True)
print(sum(A[:M//2])-sum(A[M//2:]))
