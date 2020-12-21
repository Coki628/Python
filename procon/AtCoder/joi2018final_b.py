# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/joi/2018/2019-ho/2019-ho-t2-review.pdf
　　　https://drken1215.hatenablog.com/entry/2019/02/15/014400
・自力ならず。。
・ソートして貪欲
・制約10万ならそういう系の方針だろうとは思ったけどうまく基準を見つけられなかった。
　絵を基準にしてたり小さい方から見たりしてた。
・正しいのは大きい方からやるのと、枠を基準にして合う絵を探すこと。
・言われればああ、ってなるけど、自分でこれが正しい、って導くの、まじ難しいんだよな。
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
VS = []
for i in range(N):
    s, v = MAP()
    VS.append((v, s))
C = [0] * M
for i in range(M):
    C[i] = INT()

# 絵は価値の降順ソート
VS.sort(reverse=1)
# 枠は大きさの降順ソート
C.sort(reverse=1)

# 大きい枠から順に当てはまる絵があったら貪欲に割り当てる
i = j = 0
ans = 0
while i < M and j < N:
    c = C[i]
    while j < N:
        v, s = VS[j]
        if s <= c:
            ans += 1
            j += 1
            break
        j += 1
    i += 1
print(ans)
