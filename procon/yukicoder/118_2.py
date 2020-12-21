"""
参考：http://rsujskf.s602.xrea.com/?yukicoder_118
・色々考えなくても、普通に3重ループにするだけでもよかった。。
"""

import sys
from collections import Counter

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

N = INT()
A = LIST()

C = sorted(Counter(A).items())
M = len(C)
ans = 0
for i, (k1, v1) in enumerate(C):
    for j, (k2, v2) in enumerate(C[i+1:], i+1):
        for k, (k3, v3) in enumerate(C[j+1:], j+1):
            ans += v1 * v2 * v3
            ans %= MOD
print(ans)
