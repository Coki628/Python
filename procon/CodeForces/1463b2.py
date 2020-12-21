"""
参考：みんなのツイート
・別解。TL見てても、1交互派とこっちのMSB派が半々くらいな感じがした。
・MSB(最上位ビット)
・全部2冪にすると考えると、隣り合う数が割り切れる方は、そもそも素因数が2しかないので全て問題なくなる。
　で、総和が半分以下の方も、最上位ビットを残せば元の数の半分以下になることが絶対にないのでこれもOK。
　よって数列の全ての数について、最上位ビットだけ残すようにすればいい。
・ああなるほどー、これ頭いいなー、って思うけど、こういう全く方針の違う解法が2つある時って、
　自分が思いつかなかった方の解法の方が頭いいなーっていつも思うものだよね。
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

for _ in range(INT()):
    N = INT()
    A = LIST()

    ans = [0] * N
    for i in range(N):
        ans[i] = 1<<(A[i].bit_length()-1)
    print(*ans)
