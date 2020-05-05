"""
・さくっと自力AC
・ソートしてスタックでシミュレーション。隣と偶奇が異なれば消せる。
→別にシミュレーションしなくても、偶奇の数数えて差分、だけでいけるぽい。。
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
EPS = 10 ** -10

N = INT()
A = LIST()

A.sort()
stack = []
for a in A:
    if stack and stack[-1] % 2 != a % 2:
        stack.pop()
    else:
        stack.append(a)
ans = len(stack)
print(ans)
