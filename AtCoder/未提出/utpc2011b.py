# -*- coding: utf-8 -*-

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

S = input()
N = len(S)

cnt = 0
for i in range(N//2):
    j = N - i - 1
    # ペアが成立する4つを列挙、これら以外なら要変更
    if not S[i] == S[j] == 'w' and not S[i] == S[j] == 'i' \
            and not (S[i] == '(' and S[j] == ')') \
            and not (S[i] == ')' and S[j] == '('):
        cnt += 1
if N % 2 == 1:
    if S[N//2] != 'i' and S[N//2] != 'w':
        cnt += 1
print(cnt)
