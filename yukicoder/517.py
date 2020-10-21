"""
・自力AC！
・文字列をグラフに帰着、DAG
・残っている断片を手がかりとして辺を張って、DAGのメモ化再帰みたいにやる。
・一番長くなったところが元の長さLと一致していればそれを出力。そうでなければNG。
・公式解は貪欲を紹介してたけど、これ含め色々解法はありそう。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = 26
L = 0
nxt = [-1] * N
used = [0] * N
for _ in range(2):
    for _ in range(INT()):
        li = [ord(s)-65 for s in input()]
        L += len(li)
        for i in range(len(li)):
            if i < len(li)-1:
                a, b = li[i], li[i+1]
                nxt[a] = b
            used[li[i]] = 1
L //= 2

memo = [''] * N
def dfs(u):
    if not used[u]:
        return ''
    if memo[u]:
        return memo[u]
    res = ''
    if nxt[u] != -1:
        res += dfs(nxt[u])
    res += chr(u+65)
    memo[u] = res
    return res

ans = ''
for i in range(N):
    res = dfs(i)
    if len(res) > len(ans):
        ans = res
if len(ans) == L:
    print(ans[::-1])
else:
    print(-1)
