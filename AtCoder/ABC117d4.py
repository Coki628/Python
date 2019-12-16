# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
・XOR系、各ビット独立で考えられる。
・NとM間違えて無限WAしてた。。それ直したら即突破。
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

N, K = MAP()
A = LIST()

M = max(A+[K]).bit_length()

bitcnt = [0] * M
for a in A:
    # ビット毎に立っている数を数える
    i = 0
    while (a>>i) > 0:
        bitcnt[i] += a >> i & 1
        i += 1

# 最大値を作るようなxの作成
x = 0
# 大きい方から貪欲に見ていく
for i in range(M-1, -1, -1):
    bit = 1 << i
    # 数列中のこのビットが立っている数が半分以下で、これを足してもxがKを超えないなら、このビットを立てる
    if bitcnt[i] <= N // 2 and x + bit <= K:
        x += bit

ans = 0
for a in A:
    ans += a ^ x
print(ans)
