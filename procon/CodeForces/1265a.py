# -*- coding: utf-8 -*-

"""
・なんとかACしたけど、条件分岐とかスパゲッティ過ぎて泣きそうになった。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    S = input()
    N = len(S)
    ans = []
    for i, s in enumerate(S):
        if s == '?':
            if N == 1:
                ans += ['a']
            elif i == 0:
                if S[i+1] == 'a':
                    ans += ['b']
                elif S[i+1] == 'b':
                    ans += ['c']
                else:
                    ans += ['a']
            elif i == N-1:                
                if S[i-1] == 'a':
                    ans += ['b']
                elif S[i-1] == 'b':
                    ans += ['c']
                elif S[i-1] == 'c':
                    ans += ['a']
                else:
                    ans += [chr((ord(ans[i-1])-97+1)%3+97)]
            else:
                if (ans[i-1] == 'a' or ans[i-1] == 'b') and (S[i+1] == 'a' or S[i+1] == 'b'):
                    ans += ['c']
                elif (ans[i-1] == 'b' or ans[i-1] == 'c') and (S[i+1] == 'b' or S[i+1] == 'c'):
                    ans += ['a']
                elif (ans[i-1] == 'c' or ans[i-1] == 'a') and (S[i+1] == 'c' or S[i+1] == 'a'):
                    ans += ['b']
                else:
                    ans += [chr((ord(ans[i-1])-97+1)%3+97)]
        else:
            ans += [S[i]]
    ans = ''.join(ans)
    for i, s in enumerate(ans[1:], 1):
        if s == S[i-1]:
            print(-1)
            break
    else:
        print(ans)
