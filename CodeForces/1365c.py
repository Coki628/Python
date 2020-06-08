"""
参考：https://twitter.com/koboshi_kyopro/status/1269673238533033984
　　　https://codeforces.com/blog/entry/78504
・自力ならず。
・各要素それぞれについて、何個ずらした時に一致するかを調べる。
・これは自力で通したかったけど、雰囲気近くて微妙にズレてる方針で走っていたため、力不足。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

N = INT()
A = [a-1 for a in LIST()]
B = [b-1 for b in LIST()]

atoi = {}
for i, a in enumerate(A):
    atoi[a] = i

C = Counter()
for j, b in enumerate(B):
    C[(atoi[b]-j)%N] += 1

ans = 0
for v in C.values():
    ans = max(ans, v)
print(ans)
