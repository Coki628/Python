"""
・マラソン、貪欲法
・コンテスト26個全部を各回で試して、一番いいものに進む。
"""

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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

D = INT()
C = LIST()
S = [[]] * D
for i in range(D):
    S[i]  = LIST()
M = 26

def check(i, score, t, last):
    score += S[i][t]
    for j in range(M):
        last[j] += 1
    last[t] = 0
    for j in range(M):
        score -= C[j] * last[j]
    return score

last = [0] * M
ans = []
score = 0
for i in range(D):
    mx = -INF
    idx = -1
    for t in range(M):
        res = check(i, score, t, last[:])
        if res > mx:
            mx = res
            idx = t
    t = idx
    score += S[i][t]
    for j in range(M):
        last[j] += 1
    last[t] = 0
    for j in range(M):
        score -= C[j] * last[j]
    ans.append(t+1)
for t in ans:
    print(t)
