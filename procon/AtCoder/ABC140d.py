# -*- coding: utf-8 -*-

"""
・400点自力AC！まー終了間際ギリギリだったけど。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

N, K = MAP()
S = input()

is_ok = [0] * N
for i in range(N):
    if S[i] == 'L':
        if i != 0 and S[i-1] == 'L':
            is_ok[i] = 1
    else:
        if i != N-1 and S[i+1] == 'R':
            is_ok[i] = 1

cur = S[0]
A = []
st = 0
tmp = is_ok[0]
for i in range(1, N):
    if S[i] == cur:
        tmp += is_ok[i]
    else:
        A.append((tmp, st, i, cur))
        st = i
        tmp = is_ok[i]
        cur = S[i]
A.append((tmp, st, N, cur))

LR = {'L': [], 'R': []}
for cnt, st, end, d in A:
    if st == 0 or end == N:
        LR[d].append(1)
    else:
        LR[d].append(2)

LR['L'].sort(reverse=True)
LR['R'].sort(reverse=True)
ans = sum(is_ok)

if ans == N-1:
    print(ans)
    exit()

ans += max(sum(LR['L'][:K]), sum(LR['R'][:K]))
print(ans)
