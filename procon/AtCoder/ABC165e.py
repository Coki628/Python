# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc165/editorial.pdf
　　　https://twitter.com/___chicken/status/1256601063370534913
・自力ならず。この発想は出ないだろー。。
・半分ずつで回る感じにして、片方は反対側に移るところで1つ余分に番号を進める。
・実際これで試すと被るとこないんだけど、この半分に分けるに至る思考のプロセスとか
　どんなんなんだろうね。1つずらすにしてもそうだけどさ。
・追記。ちょっと分かる考え方を聞けたので。
・1〜Nまでの人々は1ずつ値を増やしながら周期Nで回っているので、
　N角形(隣同士の頂点を繋いでサイクルになったグラフでもいいと思う)を考える。
　で、ここをくるくる1周回る時に同じペアを作らないように、M本の辺を張る感じが目標。
　そうなると、頂点間の距離が同じ辺を2本張ると、その2本を通る時に同じペアになってしまうことが分かる。
　よって、距離が同じの辺を2本作らないように、うまくM本の辺が張れればOK。
　これをやろうとすると、結果的に最初に書いたような作り方に行き着く。
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

N, M = MAP()

ans = list2d(M, 2, 0)
num = 1
for i in range(M//2):
    ans[i][0] = num
    num += 1
num += 1
for i in range(M//2-1, -1, -1):
    ans[i][1] = num
    num += 1
for i in range(M//2, M):
    ans[i][0] = num
    num += 1
for i in range(M-1, M//2-1, -1):
    ans[i][1] = num
    num += 1

for a, b in ans:
    print(a, b)
