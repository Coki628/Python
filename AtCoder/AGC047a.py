"""
・なんとか自力AC
・数列、ペアの個数、倍数、誤差、GCD
・10^9して小数をなくすと、ペアした時整数になるかどうかはペアした時10^18で割り切れるかにできる。
・これなら全ペアからある数(10^18)の倍数になるものを数えるっていう、ありそうな問題になる。
・ゆきこで見た気がして、まだ解いてないやつだったけど調べた。(問題990)
・で、倍数になるやつをa*bペアで数え上げるには、ある数とGCDして余計な部分を削ぎ落として、
　すると種類数がずいぶん減るから集計してまとめて数える、で2重ループが間に合う。
・ここまでの方針がそこそこの時間で整って提出、WA。。
・そこからずいぶん手こずったけど、結局整数にする時の切り捨て誤差が原因だったぽい。。
・最初Decimalで出して時間制約ギリギリAC。
　int切り捨てを単にroundに変えたらそれでもちゃんとAC、こっちは時間も余裕。
・足りない知見をコンテスト中に調べてちゃんと適用できたのは良かったな。
"""

import sys
from collections import Counter

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
MOD = 10 ** 19 + 7
EPS = 10 ** -10

def gcd(a, b):
    while b: a, b = b, a % b
    return a

N = INT()
K = 10 ** 9
L = 10 ** 18
A = [0] * N
C = Counter()
for i in range(N):
    a = float(input())
    # ここint()だと切り捨てでバグる
    A[i] = round(a * K)
    C[gcd(A[i], L)] += 1

ans = 0
for k1, v1 in C.items():
    for k2, v2 in C.items():
        if k1 == k2:
            if k1*k2 % L == 0:
                ans += v1*(v1-1) // 2
        elif k1 < k2:
            if k1*k2 % L == 0:
                ans += v1 * v2
print(ans)
