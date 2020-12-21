"""
・自力AC
・最短経路、ダイクストラ
・間の駅は重み付けして最低限の頂点だけ持って最短経路やればいい。
・方針はすぐ立ったけど、実装ちょっとだるいやつ。
　うまくループできれいにまとめたかったけど、かえってバグっても嫌なのでほぼ愚直に書いた。
・それでも唯一ループさせたとこバグらせて1ペナ。。
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

def dijkstra(nodes: list, src: int) -> list:
    """ ダイクストラ高速化版(隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    N = len(nodes)
    res = [INF] * N
    que = [src]
    res[src] = 0
    while que:
        cur = heappop(que)
        dist = cur // N
        cur %= N
        if res[cur] < dist:
            continue
        for nxt, cost in nodes[cur]:
            if dist + cost < res[nxt]:
                res[nxt] = dist + cost
                heappush(que, (dist+cost)*N+nxt)
    return res

x, y, z = MAP()
s1, t1 = input().split()
s2, t2 = input().split()
t1 = int(t1)
t2 = int(t2)

nodes = [[] for i in range(8)]

# 各路線の始点〜終点
nodes[0].append((1, x-1))
nodes[1].append((0, x-1))
nodes[2].append((3, y-1))
nodes[3].append((2, y-1))
nodes[4].append((5, z-1))
nodes[5].append((4, z-1))

# 各路線の始点同士と終点同士
for i in range(0, 6, 2):
    nodes[i].append(((i+2)%6, 1))
    nodes[(i+2)%6].append((i, 1))
    nodes[i+1].append(((i+3)%6, 1))
    nodes[(i+3)%6].append((i+1, 1))

# 出発と到着が同じ路線なら直線距離
if s1 == s2:
    nodes[6].append((7, abs(t1-t2)))
    nodes[7].append((6, abs(t1-t2)))

# 出発〜路線の始点と終点
if s1 == 'A':
    nodes[0].append((6, t1-1))
    nodes[6].append((0, t1-1))
    nodes[1].append((6, x-t1))
    nodes[6].append((1, x-t1))
elif s1 == 'B':
    nodes[2].append((6, t1-1))
    nodes[6].append((2, t1-1))
    nodes[3].append((6, y-t1))
    nodes[6].append((3, y-t1))
else:
    nodes[4].append((6, t1-1))
    nodes[6].append((4, t1-1))
    nodes[5].append((6, z-t1))
    nodes[6].append((5, z-t1))

# 到着〜路線の始点と終点
if s2 == 'A':
    nodes[0].append((7, t2-1))
    nodes[7].append((0, t2-1))
    nodes[1].append((7, x-t2))
    nodes[7].append((1, x-t2))
elif s2 == 'B':
    nodes[2].append((7, t2-1))
    nodes[7].append((2, t2-1))
    nodes[3].append((7, y-t2))
    nodes[7].append((3, y-t2))
else:
    nodes[4].append((7, t2-1))
    nodes[7].append((4, t2-1))
    nodes[5].append((7, z-t2))
    nodes[7].append((5, z-t2))

ans = dijkstra(nodes, 6)[7]
print(ans)
