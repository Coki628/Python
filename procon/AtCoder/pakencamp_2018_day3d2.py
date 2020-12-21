# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/pakencamp-2018-day3/submissions/7061332
・セグ木使わないで累積和応用だけでやる解
・今回実はマイナスの所を通る場合もmaxは0にできるので、区間和自体が負になることはなく、
　累積和の逆を辿るとそれが区間maxになってる。
・これだとpypyACの0.3秒。やっぱ累積和だけだと速い。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = [0] + LIST(N-1)

# 端からのmax取得のために累積和する(累積和した方向と反対側に向かう時に使う)
acc = [0] * N
for i in range(1, N):
    # 部屋を動かない選択ができるため、0とのmaxにする
    acc[i] = max(0, acc[i-1]+A[i])
rev_acc = [0] * N
for i in range(N-2, -1, -1):
    rev_acc[i] = max(0, rev_acc[i+1]+A[i+1])

for i in range(N):
    # 左に出る、右に出る、から最大を取る
    ans = max(acc[i], rev_acc[i])
    print(ans)
