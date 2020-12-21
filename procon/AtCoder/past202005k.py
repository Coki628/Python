"""
・親の付け替え的なことをして状態を更新して、最後は再帰探索。
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

N, Q = MAP()
top = [0] * N
par = [0] * N
for i in range(N):
    top[i] = i
    par[i] = N + i
for i in range(Q):
    f, t, x  = MAP()
    f -= 1; t -= 1; x -= 1
    tmp = top[f]
    top[f] = par[x]
    par[x] = top[t]
    top[t] = tmp

ans = [-1] * N
def rec(cur):
    if cur >= N:
        return cur - N + 1
    res = rec(par[cur])
    ans[cur] = res
    return res

for i in range(N):
    if ans[i] == -1:
        rec(i)
for a in ans:
    print(a)
