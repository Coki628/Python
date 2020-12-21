# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc147/editorial.pdf
・自力ならず。。
・XOR系、各ビット独立で考える。
・再三出てきてるポイントなだけに、全然見抜けなかった自分つらい。。
・計算量60ビットと30万Nのループで、60*30万=1800万はさすがにpythonきつくて、
　pythonTLEでpypy0.7秒。
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
A = LIST()

ans = 0
# 各ビット独立で考える
for i in range(60):
    bit = 1 << i
    cnt1 = 0
    for a in A:
        # このビットが立っている場所を数える
        if a & bit:
            cnt1 += 1
    cnt0 = N - cnt1
    # 0ビットの数 * 1ビットの数 で1になるペア数
    cnt = cnt0 * cnt1 % MOD
    ans += cnt * bit % MOD
print(ans%MOD)
