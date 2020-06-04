"""
・自力AC
・グラフ、木の走査
・色々方針ミスってちょっと時間食った。。
・結局、3本以上枝分かれしてる所は付け替えが要るからカウントする、でOK。
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

def dfs(nodes, src):
    """ DFS(木、スタック、重みなし) """

    stack = [(src, -1)]
    ans = 0
    while stack:
        u, prev = stack.pop()
        ans += max(len(nodes[u]) - 2, 0)
        for v in nodes[u]:
            if v != prev:
                stack.append((v, u))
    return ans

N = INT()
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

ans = dfs(nodes, 0)
print(ans)
