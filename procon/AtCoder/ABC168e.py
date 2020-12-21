"""
参考：https://img.atcoder.jp/abc168/editorial.pdf
　　　https://twitter.com/laycrs/status/1262018473141493760
・自力ならず。むずかった。。
・数え上げ、式変形、場合分け
・式変形より、Ai/Bi = -Bj/Aj となるペアは同居できないと分かる。
　よって上記の形で各イワシを集計。(ここまではできた)
・また数え上げパートが出なかった。。まあそれができたとしても場合分けできなかったぽいけど。。
・対になる各グループについて、どちらから取るか決めてしまえば後は自由に取る、取らないを選択できるので、
　各2グループについて、2^グループ1の要素数 + 2^グループ2の要素数 - 1 となる。
　(-1はこの2グループから何も取らない、の重複を引く分)
・これだけで終わらない。例外パターンが2つある。
・a,b両方0の場合は、1匹取る以外できないので別集計。
・a,bどちらか片方0の場合は、反対側の値や符号に関係なく同じグループとして集計。
・どちらの場合も実際に試せば確かに分かるが、コンテスト中に落ち着いて全部整理してまとめるのは大変だ。。
"""

import sys
from collections import Counter
from math import gcd

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
C = Counter()
zero = 0
for i in range(N):
    a, b = MAP()
    # 例外場合分け1
    if a == b == 0:
        zero += 1
    # 例外場合分け2
    elif a == 0:
        C[(0, 1, 0)] += 1
    elif b == 0:
        C[(1, 0, 1)] += 1
    # 通常
    else:
        g = gcd(a, b)
        ga, gb = a // g, b // g
        C[((a < 0) ^ (b < 0), abs(ga), abs(gb))] += 1

ans = 1
for k, v in C.items():
    if v == 0:
        continue
    op, a, b = k
    ans *= pow(2, C[(op, a, b)], MOD) + pow(2, C[(1-op, b, a)], MOD) - 1
    ans %= MOD
    # 対になるグループを空にしておく
    if (1-op, b, a) in C:
        C[(1-op, b, a)] = 0
# -1は「何も取らない」の分
print((zero+ans-1)%MOD)
