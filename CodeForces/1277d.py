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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = [''] * N
    org = [''] * N
    se = set()
    for i in range(N):
        S = input()
        se.add(S)
        org[i] = S
        A[i] = S[0] + S[-1]
    C = Counter(A)
    if len(C) == 1 and C['00'] > 0 or len(C) == 1 and C['11'] > 0:
        print(0)
        print()
    elif len(C) == 2 and C['00'] > 0 and C['11'] > 0:
        print(-1)
    else:
        cnt = 0
        ans = []
        if C['01'] > C['10']:
            diff = C['01'] - C['10']
            while diff >= 2:
                cnt += 1
                diff -= 2
            for i, a in enumerate(A):
                if len(ans) == cnt:
                    break
                if a == '01' and org[i][::-1] not in se:
                    ans.append(i+1)
        if C['10'] > C['01']:
            diff = C['10'] - C['01']
            while diff >= 2:
                cnt += 1
                diff -= 2
            for i, a in enumerate(A):
                if len(ans) == cnt:
                    break
                if a == '10' and org[i][::-1] not in se:
                    ans.append(i+1)
        print(len(ans))
        print(*ans)
