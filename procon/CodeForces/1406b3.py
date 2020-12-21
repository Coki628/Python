"""
・試しにABC173Eの誤差死したlogで累積和とるやつを貼ってみた。なんとこれでもAC。。
・まあ取る数5個固定だし、そんなに誤差出なそうではあったけども。。。
・微妙に問題設定違った気がしたからやらなかったんだけど、
　ここまで貼るだけで行けたならやってみればよかった。。。
"""

import sys
from itertools import accumulate
from math import log2

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
INF = 10**19
MOD = 10**9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST()
    K = 5
    
    A1 = []
    A2 = []
    B1 = []
    B2 = []
    zero = 0
    for a in A:
        if a > 0:
            A1.append(a)
            B1.append(log2(a))
        elif a < 0:
            A2.append(-a)
            B2.append(log2(-a))
        else:
            zero += 1

    A1.sort()
    A2.sort()
    B1.sort()
    B2.sort()
    acc1 = [0] + list(accumulate(B1))
    accrev1 = [0] + list(accumulate(B1[::-1]))
    acc2 = [0] + list(accumulate(B2))
    accrev2 = [0] + list(accumulate(B2[::-1]))
    mx = -INF
    mn = INF
    idx1 = (-1, -1)
    idx2 = (-1, -1)
    for i in range(K+1):
        j = K - i
        if len(B1) < i or len(B2) < j:
            continue
        if j % 2 == 0:
            sm = accrev1[i] + accrev2[j]
            if sm > mx:
                mx = sm
                idx1 = (i, j)
        else:
            sm = acc1[i] + acc2[j]
            if sm < mn:
                mn = sm
                idx2 = (i, j)

    if mx != -INF:
        ans = 1
        n, m = idx1
        A1.reverse()
        A2.reverse()
        for i in range(n):
            ans *= A1[i]
        for i in range(m):
            ans *= -A2[i]
        print(ans)
    elif zero:
        print(0)
    else:
        ans = 1
        n, m = idx2
        for i in range(n):
            ans *= A1[i]
        for i in range(m):
            ans *= -A2[i]
        print(ans)
