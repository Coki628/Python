"""
・さくっと自力AC
・ちゃんとソートがよしなにやってくれるようなkey,valueを予め割り当てておけばOK。
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

kton1 = {'D': 0, 'C': 1, 'H': 2, 'S': 3}
ntok1 = {0: 'D', 1: 'C', 2: 'H', 3: 'S'}
kton2 = {'A': 1, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}
ntok2 = {1: 'A', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'}
for i in range(2, 10):
    kton2[str(i)] = i
    ntok2[i] = str(i)

N = INT()
S = list(input().split())

ans = []
for i in range(N):
    s, n = S[i]
    ans.append((kton1[s], kton2[n]))

ans.sort()
for i in range(N):
    s, n = ans[i]
    ans[i] = ntok1[s] + ntok2[n]
print(*ans)
