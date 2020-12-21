"""
参考：https://twitter.com/laycrs/status/1300109807874011138
・自力ならず。。悔しいなー今回これ通せれば結構爆上げっぽかったのに。
・互いに素、MOD
・操作回数3回ってとこから、全体と全体-1を使いそうな気がするのは考えたのに、
　正解の方針に至れなかった。考察力不足。。。
・1回目の操作は全要素に対して、Nをいくつか足すことにする。
　mod (N-1) で考えた時、Nは必ず1になる。すると、Nを足した数だけ、
　元の数はmod (N-1) で+1されていくことが分かる。
・使える数の最大値は十分にあるので、これを使って元の数がmod (N-1) で0になるところまでNを足せば、
　全ての値がmod (N-1) で0、つまりN-1の倍数になっているので、次の操作でN-1個については全部0にできる。
・残った1個は3回目の操作で0にできる。
・この辺はちゃんと考察したかったなぁ、、悔しい。
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
A = LIST()

# 要素1個だとN-1がないので例外処理
if N == 1:
    print(1, 1)
    print(-A[0])
    print(1, 1)
    print(0)
    print(1, 1)
    print(0)
    exit()

ans1 = [0] * N
print(1, N)
for i in range(N):
    need = -A[i] % (N-1)
    ans1[i] = N * need
print(*ans1)

ans2 = [0] * (N-1)
print(1, N-1)
for i in range(N-1):
    ans2[i] = -(A[i]+ans1[i])
print(*ans2)

print(N, N)
print(-(A[-1]+ans1[-1]))
