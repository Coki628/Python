# -*- coding: utf-8 -*-

"""
参考：https://www.youtube.com/watch?v=7IlBVSglZqc
　　　https://img.atcoder.jp/abc146/editorial.pdf
　　　https://maspypy.com/atcoder-%e5%8f%82%e5%8a%a0%e6%84%9f%e6%83%b3-2019-11-16abc-146
・これはさっぱり解法出ず。
・累積和、式変形、度数分布
・区間和が0になる区間数だったら、acc[l-1] - acc[r] == 0 つまり累積和取って acc[l-1] == acc[r] になる区間を数えればいい。
・よって、区間和がmod Kで0になるような区間数、なら出る。
・式変形：sum(ある区間) % K == len → (sum(ある区間) % K) - len == 0
・つまり全ての区間から区間長を引く、これは全要素から-1すれば実現できる。
・それが分かれば、後はZero-Sum Ranges(AGC023a)の類題。
　出現済の数を集計しながら、自分より前に出現した自分と同じ値を足していく。
・ただし、長さK以上はダメなので、K以上前の値は減らしていく必要がある。
"""

import sys
from collections import defaultdict
from itertools import accumulate

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
# 全要素-1した状態で作る
A = [(a-1)%K for a in LIST()]

# 累積和
acc = [0] + list(accumulate(A))
acc = [a%K for a in acc]

cnt = defaultdict(int)
ans = 0
for i, a in enumerate(acc):
    # 自分より前に出現した自分と同じ値、を答えに足す
    ans += cnt[a]
    # 度数分布に自分を加える
    cnt[a] += 1
    # K個以上前の値は減らす
    if i - K + 1 >= 0:
        cnt[acc[i-K+1]] -= 1
print(ans)
