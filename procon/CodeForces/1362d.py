"""
参考：https://twitter.com/50j1_/status/1268590927657922560
　　　https://codeforces.com/contest/1361/submission/82501656
・自力ならず。BFSしたり頑張ってみたけど方針からダメだったね。。
・グラフ、mex、入出力
・要点は、頂点iに隣接する頂点から参照できる値で、自分の欲しい値A[i]をmexとできるかどうか。
・それぞれの隣接する頂点を確認して、mexを更新できるなら更新。最終的にmex[i]がA[i]以上になっていれば、
　書き込み順を適切に並べればA[i]にできることになる。逆にA[i]未満だとどうやってもその値にできない。
・あとそれとは別に、隣接する頂点で同じA[i]を持つものがないかもチェックする。
・両方の条件を満たしていれば、後はA[i]の小さい方からソートして並べればそれが適切。
・頂点、辺ともに最大50万と制約がきつめ。入力をbufferの方にしたらギリギリ通った。pypyAC1.996秒。。
・ソートにitemgetter付けたら少しマシになった。でもこれは誤差の範囲かな。pypyAC1.918秒。
"""

import sys
from operator import itemgetter

def input(): return sys.stdin.buffer.readline().strip()
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

N, M = MAP()
edges = []
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    edges.append((a, b))
A = [a-1 for a in LIST()]

mex = [0] * N
se = [set() for i in range(N)]
for a, b in edges:
    # 隣接頂点でAの値が同じのがあったら即NG
    if A[a] == A[b]:
        print(-1)
        exit()
    # 隣接頂点に初めて使われる値だったら集合に追加
    if A[b] not in se[a]:
        se[a].add(A[b])
        # 今回更新した値がmexだったらmexも更新する
        if mex[a] == A[b]:
            x = mex[a]
            # まだ使われてない最小まで増やす
            while x in se[a]:
                x += 1
            mex[a] = x
    if A[a] not in se[b]:
        se[b].add(A[a])
        if mex[b] == A[a]:
            x = mex[b]
            while x in se[b]:
                x += 1
            mex[b] = x

for u in range(N):
    # mexが、頂点uが達成したい値A[u]未満だと無理
    if mex[u] < A[u]:
        print(-1)
        exit()

# 上の2つの条件を満たしていたら、Aの値でソートして並べれば適切になる
AI = [(a, i) for i, a in enumerate(A)]
AI.sort(key=itemgetter(0))
ans = []
for a, i in AI:
    ans.append(i+1)
print(*ans)
