# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc143/editorial.pdf
　　　http://drken1215.hatenablog.com/entry/2019/10/20/032700
・2点固定して3重ループを2重にする系
・二分探索
・ほぼ同じことを考えたんだけど、各にぶたんで重複して数えてしまう気がしたり、
　同値をうまく考慮しないと正しく数えられない気がしたりでまとめられなかった。
・結果的に、ソートして小さい2つって決めれば、3つ目は2番目より先だけ考えればいいって話で、
　全然複雑なことはなかった。。
・これは自力通したかったよなぁ。。。
・計算量2000のNでN^2logNだから400万*22くらいで8800万、、絶対pythonつらいと思ったけど1.3秒AC。
・やっぱbisectが内部的にかなり速いんだろうか。
・ちなみにC++はN^3に枝刈りで通った。
"""

import sys
from bisect import bisect_left

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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

A.sort()
ans = 0
for i in range(N):
    a = A[i]
    for j in range(i+1, N):
        # 小さい方2本a,bを決めておく(a < b)
        b = A[j]
        # 条件を満たすcの範囲は b < c < a+b の間
        k = bisect_left(A, a+b)
        ans += k - (j+1)
print(ans)
