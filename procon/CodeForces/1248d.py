# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/70720
・括弧系
・"("で+1、")"で-1して、最小値になった回数が、括弧のグループ数になる。
・これはTLE
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
INF = float('inf')
MOD = 10 ** 9 + 7

N =  INT()
S = list(input())

C = Counter(S)
if C['('] != C[')']:
    print(0)
    print(1, 1)
    exit()

# 最小値になった回数を返す
def check(S):
    cnt = 0
    C = Counter()
    for s in S:
        if s == '(':
            cnt += 1
        else:
            cnt -= 1
        C[cnt] += 1
    C = sorted(C.items())
    return C[0][1]

mx = 0
ans = (1, 1)
for i in range(N):
    for j in range(i+1, N):
        S[i], S[j] = S[j], S[i]
        cnt = check(S)
        if cnt > mx:
            mx = cnt
            ans = (i+1, j+1)
        S[i], S[j] = S[j], S[i]
print(mx)
print(*ans)
