# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC
・後ろから貪欲、辞書順最小
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

S = input()
T = input()
N = len(S)
M = len(T)

def check(S, T):
    N = len(S)
    for i in range(N):
        if S[i] != '?' and S[i] != T[i]:
            return False
    return True

# 前の方はなるべくaで埋めたいので、後ろからTとの一致を探す
ans = [''] * N
for i in range(N-M, -1, -1):
    # 当てはまる部分列を見つけたら、答えに追記してbreak
    if check(S[i:i+M], T):
        for j in range(i, i+M):
            ans[j] = T[j-i]
        break
else:
    # 見つからなければNG
    print('UNRESTORABLE')
    exit()

for i in range(N):
    if S[i] != '?':
        ans[i] = S[i]
    # 辞書順最小にしたいので、余っている?はaで埋める
    elif ans[i] == '':
        ans[i] = 'a'
print(''.join(ans))
