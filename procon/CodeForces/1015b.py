# -*- coding: utf-8 -*-

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def check(C1, C2):
    for k, v in C1.items():
        if C2[k] != v:
            return False
    return True

N = INT()
S = list(input())
T = list(input())

C1 = Counter(S)
C2 = Counter(T)
if not check(C1, C2):
    print(-1)
    exit()

ans = []
for i in range(N):
    if S[i] != T[i]:
        for j in range(i+1, N):
            if S[j] == T[i]:
                while i != j:
                    S[j-1], S[j] = S[j], S[j-1]
                    j -= 1
                    ans.append(j+1)
                break
print(len(ans))
print(*ans)
