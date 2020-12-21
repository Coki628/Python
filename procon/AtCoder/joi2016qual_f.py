# -*- coding: utf-8 -*-

"""
・自力AC！
・ダイクストラ応用
・普通のダイクストラから、条件に合わせて状態(添字)を増やして管理する。
・こういうのやると、ダイクストラも詰まるところDPの一種ってのを実感するな。
・1万*200*2で400万頂点あるような感じだったから計算量心配だったけど、
　pythonAC2.1秒、pypyAC1.2秒。
・ダイクストラの計算量は頂点より辺依存のが大きいから、
　頂点だけ増える分には100万単位でも何とかいける感じなのかな。
・と思ったけど公式解説の計算量見たら辺も増えたことになってるな。どうなんだろー。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

def dijkstra(nodes: list, src: int) -> list:
    """ ダイクストラ(隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    N = len(nodes)
    # res[i][j][k] := 頂点iにいて、寒いか暑い部屋を出てからj分経過して、最後にいたのが(快適除く)寒いor暑い部屋(k=0,1)の時の最小値
    res = list3d(N, X+1, 2, INF)
    que = [(0, src, X, 0)]
    res[src][X][0] = 0
    while que:
        dist, cur, x, t = heappop(que)
        if res[cur][x][t] < dist:
            continue
        for nxt, cost in nodes[cur]:
            # 快適な部屋へ
            if T[nxt] == 1:
                if dist + cost < res[nxt][max(x-cost, 0)][t]:
                    res[nxt][max(x-cost, 0)][t] = dist + cost
                    heappush(que, (dist+cost, nxt, max(x-cost, 0), t))
            # 寒い部屋へ
            elif T[nxt] == 0:
                # 直前(快適除く)は寒い部屋
                if t == 0:
                    if dist + cost < res[nxt][X][0]:
                        res[nxt][X][0] = dist + cost
                        heappush(que, (dist+cost, nxt, X, 0))
                # 直前(快適除く)は暑い部屋
                else:
                    if x-cost > 0:
                        continue
                    else:
                        if dist + cost < res[nxt][X][0]:
                            res[nxt][X][0] = dist + cost
                            heappush(que, (dist+cost, nxt, X, 0))
            # 暑い部屋へ
            else:
                # 直前(快適除く)は寒い部屋
                if t == 0:
                    if x-cost > 0:
                        continue
                    else:
                        if dist + cost < res[nxt][X][1]:
                            res[nxt][X][1] = dist + cost
                            heappush(que, (dist+cost, nxt, X, 1))
                # 直前(快適除く)は暑い部屋
                else:
                    if dist + cost < res[nxt][X][1]:
                        res[nxt][X][1] = dist + cost
                        heappush(que, (dist+cost, nxt, X, 1))
    return res

N, M, X = MAP()
T = LIST(N)
nodes = [[] for i in range(N)]
for _ in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))

res = dijkstra(nodes, 0)
ans = INF
for i in range(X+1):
    for j in range(2):
        ans = min(ans, res[N-1][i][j])
print(ans)
