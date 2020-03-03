# -*- coding: utf-8 -*-

"""
・なんとか自力AC。
・文字列で連続する部分列数える系
・でも普通にやると100万の文字列はきついので、連続するIOIO..I毎にまとめることにした。
・後は離れた場所のそれぞれはお互いに影響を与えないので、独立に数えればOK。
・でもそのまとめるのが、結構ゴニョゴニョして大変だった。
・公式解より、KMP法とかのアルゴリズムがあれば殴れるそうだ。そのうちやってみよう。
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

N = INT()
M = INT()
S = input()

i = 0
A = []
while i+1 < M:
    if S[i] == 'I' and S[i+1] == 'O':
        cnt = 0
        while i+3 < M and S[i+2] == 'I' and S[i+3] == 'O':
            i += 2
            cnt += 1
        if i+2 == M:
            i += 1
        elif S[i+2] == 'O':
            i += 2
        elif S[i+2] == 'I':
            i += 2
            cnt += 1
        # 連続するIOI毎にまとめる
        A.append(cnt)
    else:
        i += 1
ans = 0
for a in A:
    # まとまり毎にいくつ作れるか数える
    if a >= N:
        ans += a - N + 1
print(ans)
