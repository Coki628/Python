"""
参考：http://sugarknri.hatenablog.com/entry/2016/06/10/015311
・範囲を絞って全探索
・公式解の方針。これだと素因数分解しないで全探索できる。
・よく見る3重ループを式変形して2重にするやつの和じゃなくて積みたいな感じ。
・で、積を3つに分けると、どれか1つは必ず3乗根以下になることは利用して、ループを短くするのがポイント。
・なんか普通にやったらWAが残って、sqrtとかでint()で切り捨てたとこをround()にしたら通った。誤差さん。。
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
sys.setrecursionlimit(10**9)
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

N = INT()

mn = INF
for i in range(1, round(pow(N, 1/3))+1):
    if N % i != 0:
        continue
    jk = N // i
    for j in range(i, round(pow(jk, 1/2))+1):
        if jk % j != 0 and jk // j != 0:
            continue
        k = jk // j
        mn = min(mn, i+j+k-3)
print(mn, N-1)
