"""
・最初に書いてた方の方針。こっちも同じ例外処理入れたら通った。。
"""

import sys
from string import ascii_lowercase

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    S = input()
    N = len(S)

    if S > 'atcoder':
        print(0)
        continue

    ans = -1
    T = ascii_lowercase[1:]
    for i in range(N):
        if S[i] in T:
            ans = i
            break
    if N >= 1 and S[:1] == 'a':
        T = 'uvwxyz'
        for i in range(1, N):
            if S[i] in T:
                ans = min(ans, i - 1)
                break
    if N >= 2 and S[:2] == 'at':
        T = ascii_lowercase[3:]
        for i in range(2, N):
            if S[i] in T:
                ans = min(ans, i - 2)
                break
    if N >= 3 and S[:3] == 'atc':
        T = 'pqrstuvwxyz'
        for i in range(3, N):
            if S[i] in T:
                ans = min(ans, i - 3)
                break
    if N >= 4 and S[:4] == 'atco':
        T = ascii_lowercase[4:]
        for i in range(4, N):
            if S[i] in T:
                ans = min(ans, i - 4)
                break
    if N >= 5 and S[:5] == 'atcod':
        T = ascii_lowercase[5:]
        for i in range(5, N):
            if S[i] in T:
                ans = min(ans, i - 5)
                break
    if N >= 6 and S[:6] == 'atcode':
        T = 'stuvwxyz'
        for i in range(6, N):
            if S[i] in T:
                ans = min(ans, i - 6)
                break
    print(ans)
