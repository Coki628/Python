# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc015
・グラフに帰着
・DAGじゃない普通の双方向なグラフだった。
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
words = set()
LMS = []
for i in range(N):
    lg, m, sm = input().split()
    m = int(m)
    words.add(lg)
    words.add(sm)
    LMS.append((lg, m, sm))

# 単位と頂点番号のマッピング
tonum = {}
tostr = {}
for i, w in enumerate(list(words)):
    tonum[w] = i
    tostr[i] = w

# ここからNは頂点数
N = len(words)
# 単位間を相互に変換する重み付き有向グラフを構築する
nodes = [[] for i in range(N)]
for lg, m, sm in LMS:
    u = tonum[lg]
    v = tonum[sm]
    nodes[u].append((v, m))
    nodes[v].append((u, 1/m))

# DFS(情報に矛盾がない保証がされているので、どこから辿っても各1回見ればOK)
memo = [-1] * N
stack = [(0, 1)]
while stack:
    u, c1 = stack.pop()
    memo[u] = c1
    for v, c2 in nodes[u]:
        if memo[v] == -1:
            stack.append((v, c1*c2))

# 最小単位と最大単位の値と頂点番号を取得
mn = max(memo)
mni = memo.index(mn)
mx = min(memo)
mxi = memo.index(mx)

# 最小単位で表した最大単位1の値を求める
mn *= 1 / mx
print('1{0}={1}{2}'.format(tostr[mxi], round(mn), tostr[mni]))
