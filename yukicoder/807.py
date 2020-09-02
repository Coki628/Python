"""
・自力AC
・いわゆる拡張ダイクストラ
・状態として、チケットを使う前と使った後を持ってダイクストラする。
・10万頂点20万辺で、さらに拡張やって増えてるので計算量きつめ。
　制約4秒あったのでとりあえず高速化の工夫はせずに普通に投げたけど、AC2.89秒。
・制約2秒だったら、キューに入れるのタプルじゃなくて値にするとか、
　距離格納する配列は2次元じゃなくて1次元2つにするとかやるところ。
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
    """ ダイクストラ(隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    N = len(nodes)
    res = list2d(2, N, INF)
    que = [(0, 0, src)]
    res[0][src] = 0
    while que:
        dist, flag, cur = heappop(que)
        if res[flag][cur] < dist:
            continue
        for nxt, cost in nodes[cur]:
            # 普通に進む遷移
            if dist + cost < res[flag][nxt]:
                res[flag][nxt] = dist + cost
                heappush(que, (dist+cost, flag, nxt))
            # チケットを使う遷移
            if not flag and dist < res[1][nxt]:
                res[1][nxt] = dist
                heappush(que, (dist, 1, nxt))
    return res

N, M = MAP()
nodes = [[] for i in range(N)]
for _ in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))

res = dijkstra(nodes, 0)
for i in range(N):
    ans = min(res[0][i]*2, res[0][i]+res[1][i])
    print(ans)
