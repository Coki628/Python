"""
参考：https://codeforces.com/blog/entry/82643
・何が正解か確かめるために公式解説を読んだ。
・まず、動かせるとこ全部降順ソート、でOKだった。
・これが正しいことは、2点スワップのみで考えた時、小・大を大・小に交換した時、
　前の方は大になった分大きくなっているのに対して、後の方は累積では同値であり、
　つまり減らないので、この交換は全く損しないことから分かる。
　よってこのスワップを繰り返して降順ソート状態にするのが最善、となる。
・すごいでかい負数が後ろにあると、そこが足引っ張る気がしてしまったんだけど、
　累積なので、どう並んでも結局そこに来た時には同じ値になるんだよね。。。確かに。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    A = LIST()
    B = LIST()

    ans = [0] * N
    A2 = []
    for i in range(N):
        if B[i]:
            ans[i] = A[i]
        else:
            A2.append(A[i])

    A2.sort(reverse=1)
    j = 0
    for i in range(N):
        if not B[i]:
            ans[i] = A2[j]
            j += 1
    print(*ans)
