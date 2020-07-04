"""
参考：ツイート
・自力ならず。。
・スタック
・一発考察はきつくても、こっちのスタック解法は出るべきだったなー。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST()

    stack = [A[0]]
    for i in range(1, N):
        while len(stack) >= 2 and stack[-1] < A[i]:
            stack.pop()
        if stack[-1] > A[i]:
            stack.append(A[i])
    if len(stack) == 1:
        YES()
    else:
        NO()
