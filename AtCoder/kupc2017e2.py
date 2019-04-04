# -*- coding: utf-8 -*-

"""
参考：https://ikatakos.com/pot/programming_algorithm/graph_theory/bipartite_matching
・二部最大マッチング
・重み付きだけど、片側(宝箱)にしか依らないので、ソートして普通の二部最大マッチングでいける。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

# XとYの二部グラフの最大マッチング X={0,1,2,...|X|-1} Y={0,1,2,...,|Y|-1}
#   edges[x]: xとつながるYの頂点のset
#   matched[y]: yとマッチングされたXの頂点(暫定)
 
def dfs(v, visited):
    """
    :param v: X側の未マッチングの頂点の1つ
    :param visited: 空のsetを渡す（外部からの呼び出し時）
    :return: 増大路が見つかればTrue
    """
    for u in edges[v]:
        if u in visited:
            continue
        visited.add(u)
        if matched[u] == -1 or dfs(matched[u], visited):
            matched[u] = v
            return True
    return False

N,M=MAP()
V=LIST()
# [0]:宝の価値, [1]:その宝を開けれる鍵リスト
V=[[V[i], []] for i in range(N)]

for i in range(M):
    x,y=MAP()
    V[x-1][1].append(i)
    V[y-1][1].append(i)
# 価値が高い方から決められるようにソートしておく
V.sort(reverse=True)

# グラフ読み取り
edges = [set() for _ in range(N)]
matched = [-1] * M
for i in range(N):
    for v in V[i][1]:
        edges[i].add(v)

ans=0
for i in range(N):
	# 増大路発見に成功したらTrue(=1)。合計することでマッチング数となる
	if dfs(i, set()):
		# 今回はペアが決まった宝の分の価値を足していく
		ans+=V[i][0]
print(ans)
