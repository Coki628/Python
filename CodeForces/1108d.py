# -*- coding: utf-8 -*-

"""
・自力AC
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = int(input())
S = list(input())

ans = INF
colors = 'RGB'
cnt = 0
for i in range(1, len(S)-1):
    if S[i-1] == S[i] and S[i] != S[i+1]:
        s = colors.replace(S[i-1], '').replace(S[i+1], '')
        cnt += 1
    elif S[i-1] == S[i] == S[i+1]:
        if S[i] == 'R': s = 'G'
        elif S[i] == 'G': s = 'B'
        elif S[i] == 'B': s = 'R'
        cnt += 1
    else:
        s = S[i]
    S[i] = s
if N >= 2 and S[-2] == S[-1]:
    if S[-1] == 'R': s = 'G'
    elif S[-1] == 'G': s = 'B'
    elif S[-1] == 'B': s = 'R'
    cnt += 1
    S[-1] = s
print(cnt)
print(''.join(S))
