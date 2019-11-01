# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=WpUkC4-hyLE
・蟻本演習3-2-3
・反転(ライツアウト)
・PDFいまいち理解進まなくて、解説動画のやり方でやった。
・半分までであれば、1つずらして操作することで任意の一点を自由に変更できる。
・Kが大きくなる程、真ん中周辺から自由が効かなくなるので、そこをチェックしていく。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

S = list(input())
N = len(S)

ans = ceil(N, 2)
for i in range(ceil(N, 2), N):
    # 真ん中から始めて、反対側と一致してるかと、1つ前と一致しているか
    if not S[i] == S[N-1-i] == S[i-1]:
        break
    ans += 1
print(ans)
