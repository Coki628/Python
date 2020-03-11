# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/hitachi2020/editorial.pdf
・自力無理だったー。。
・木、二部グラフ
・考察パート強い。。実装も地味に順列の振り分けどうやるかちょっと悩んだけど。
・まず、距離3のペアがどうなるかについてまとめられなかったので、スタートラインにも立てなかった感じ。。
・木は二部グラフでもあるので、距離の偶奇は、二部グラフでどっち側にいるか、に対応してる。
　距離3なら二部グラフ上では絶対反対側になる。
・次に、和または積が3、については、mod3で考えると、実はダメなペアは9通りのうち1mod3同士と2mod3同士だけだと分かる。
・最後に場合分けで、グラフの偏りが大きい時は少ない方に0mod3を全部寄せ、
　小さい時は1mod3と2mod3が両側に全部分かれるように配置する。
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

def dfs(nodes, src):
    """ DFS(木、スタック、重みなし) """

    N = len(nodes)
    stack = [(src, -1, 0)]
    dist = [-1] * N
    while stack:
        u, prev, c = stack.pop()
        dist[u] = c
        for v in nodes[u]:
            if v != prev:
                # 0,1を交互に入れて二部グラフを塗り分け
                stack.append((v, u, 1-c))
    return dist

N = INT()
nodes = [[] for i in range(N)]
for _ in range(N-1):
    a, b = MAP()
    a -= 1; b-= 1
    nodes[a].append(b)
    nodes[b].append(a)

res = dfs(nodes, 0)
cnt0 = res.count(0)
cnt1 = res.count(1)

ans = [0] * N
# グラフの偏りによって場合分け
if cnt0 <= N // 3 or cnt1 <= N // 3:
    if cnt0 > cnt1:
        cnt0, cnt1 = cnt1, cnt0
        res = [1-c for c in res]
    # 少ない方全部に0mod3入れて、多い方は残りを適当に入れる
    p1 = 1
    p2 = 2
    p3 = 3
    for i in range(N):
        if res[i] == 0:
            ans[i] = p3
            p3 += 3
        else:
            if p1 <= N:
                ans[i] = p1
                p1 += 3
            elif p2 <= N:
                ans[i] = p2
                p2 += 3
            else:
                ans[i] = p3
                p3 += 3

else:
    # それぞれに1mod3と2mod3を優先して入れて、なくなったら残りには0mod3を入れる
    p1 = 1
    p2 = 2
    p3 = 3
    for i in range(N):
        if res[i] == 0:
            if p1 <= N:
                ans[i] = p1
                p1 += 3
            else:
                ans[i] = p3
                p3 += 3
        else:
            if p2 <= N:
                ans[i] = p2
                p2 += 3
            else:
                ans[i] = p3
                p3 += 3
print(*ans)
