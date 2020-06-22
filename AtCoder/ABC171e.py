"""
・通せたけど遅かった。てかみんな速くない？確かにゆるめのEなのは分かるけど茶diffとは。。
・観察すると、自分以外の全部でXORとれば自分が出てくると気づく。
・で、前後から累積xorで前計算しておけば各答えはO(1)で出せる、と。
・ってやったんだけど、これ先に全部累積xorしておけば、それと各要素合わせれば結果的に
　自分のだけが残るからそれが答え、だね。前後から累積前計算やってそれぞれ前後から取るのは
　結構工夫ありだと思ったけど、全部累積だけで出せるならまあ無限人通すのもしょうがないか。。。
"""

import sys
from itertools import accumulate
from operator import xor

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

N = INT()
A = [0] + LIST() + [0]

acc1 = list(accumulate(A, xor))
acc2 = list(accumulate(A[::-1], xor))[::-1]
ans = [0] * (N+1)
for i in range(1, N+1):
    ans[i] = acc1[i-1] ^ acc2[i+1]
print(*ans[1:])
