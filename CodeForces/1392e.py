"""
・終了後、制約読み違いが発覚。。
・インタラクティブ、グリッド、2冪
・使っていい最大値が10^16なのに、2^16な気でいてしまって、
　移動回数25*2くらいあるのに16個しか値使えないんじゃ2冪は無理だな、ってなってた。。
・これが50個くらい使えると分かったら方針はだいたい立った。
・各移動について2冪のどっかの値を使う。同じタイミングの移動で動きうる場所は2つしかないので、
　その動きうる2つで被らないようにしつつ、同じタイミングで行く場所では同じ2冪を使い回す。
・すると、斜めに0と2冪を交互に置いていくような配置がいい、となる。(自分ノート参照)
・これが作れたら、あとは各回の移動で「行の偶奇」と「その回の値を取ったか」をXORすれば
　右に行ったか下に行ったか分かる。
・制約ちゃんと分かってれば自力ACはできたかもなー。。時間内は間に合わなかった可能性が高いけど。。
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

N = INT()

grid = list2d(N, N, 0)
k = 0
for j in range(1, N*2-1):
    i = 0
    pt = 1
    while i < N and j >= 0:
        if j < N:
            if pt:
                grid[i][j] = 1 << k
        pt = 1 - pt
        j -= 1
        i += 1
    k += 1

for i in range(N):
    print(*grid[i], flush=1)

Q = INT()
M = N*2 - 2
for _ in range(Q):
    K = INT()

    h = w = 1
    ans = [(h, w)]
    for i in range(M):
        if (h % 2 == 0) ^ (K>>i & 1):
            w += 1
        else:
            h += 1
        ans.append((h, w))
    for a in ans:
        print(*a, flush=1)
