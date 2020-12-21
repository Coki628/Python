"""
・なんとか自力AC
・ゲーム、木の走査
・しょうもない凡ミスとか勘違いで時間をかけた。。
・子孫ノードの数数えたけど、実は例外処理の後は頂点数Nの偶奇だけでよかったというギャグ問。。
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

def rec(u, prev):
    res = 0
    for v in nodes[u]:
        if v == prev:
            continue
        res += rec(v, u)
    cnt[u] = res + 1
    return res + 1

for _ in range(INT()):
    N, K = MAP()
    K -= 1
    nodes = [[] for i in range(N)]
    for i in range(N-1):
        a, b = MAP()
        a -= 1; b -= 1
        nodes[a].append(b)
        nodes[b].append(a)

    if len(nodes[K]) <= 1:
        print('Ayush')
        continue

    cnt = [0] * N
    rec(K, -1)
    ans = 0
    for v in nodes[K]:
        ans += cnt[v]
    if ans % 2 == 1:
        print('Ayush')
    else:
        print('Ashish')
