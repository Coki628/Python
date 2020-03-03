# -*- coding: utf-8 -*-

"""
・なんとか自力AC
・スタック
・スタック解いつも解けない印象があったから、ちゃんと自力で通せたのは良かった。
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
A = LIST(N)

stack = []
for i, a in enumerate(A):
    if i == 0 or A[i-1] != A[i]:
        # 色が変わる所でindexと色を持つ
        stack.append([i, a])
        if i % 2 != 0:
            # 今回と違う色が続く限り色を置き換えながら遡る
            while len(stack) >= 2 and stack[-2][1] != stack[-1][1]:
                stack[-2][1] = stack[-1][1]
                stack.pop()
M = len(stack)
ans = 0
for i in range(M-1):
    # 色が白の区間をカウント
    if stack[i][1] == 0:
        ans += stack[i+1][0] - stack[i][0]
if stack[-1][1] == 0:
    ans += N - stack[-1][0]
print(ans)
