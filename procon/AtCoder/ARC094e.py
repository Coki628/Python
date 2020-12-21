# -*- coding: utf-8 -*-

"""
・自力AC
・ゲーム系
・頑張って考えてはいるものの、雰囲気でやってみた貪欲で通ってしまうと、喜んでいいのか複雑な気持ちになる。
　とはいえまあだいたい想定解とやってることは同じだったぽいし自分なりの解釈を書いておく。
・最初から全部一致でない限り、先手はうまくやって最後のひと山しか残ってない状態に持っていける。
　逆に総和が一致している以上、最後のひと山は他が全部0になる時には絶対一致してしまう。
・で、どの山を残すのがいいかとなると、まずAのが少ない山は減らす程Bとの差を引き伸ばせるから
　率先して減らした方がいい。で、Aのが多い山の中でも、Bが少ない山を残す方が、
　最終的にそこが一致してしまうまでのターンを引き伸ばせるのでおいしい。
・よって、A > B な山の中でBが一番少ない所を最後に残すつもりで動くといい。
　答えとなる、引き伸ばせる最大ターン数は 総和 - その残す山のB となる。
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
A = [0] * N
B = [0] * N
for i in range(N):
    A[i], B[i] = MAP()

sm = 0
mn = INF
for i in range(N):
    a, b = A[i], B[i]
    if a > b:
        mn = min(mn, b)
    sm += b
if mn == INF:
    print(0)
    exit()
ans = sm - mn
print(ans)
