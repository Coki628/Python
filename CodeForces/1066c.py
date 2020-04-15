# -*- coding: utf-8 -*-

"""
・pypyTLEなのでC++でAC。
・区間加算BIT
・絶対もっとすんなり賢く方法あるんだろうなー、って思いながらも
　全体に1足す、を高速にしたかったので区間加算BITで殴った。
・左に詰める時は全体のindexを1右にずらす。
・これができたら後は各IDのindexを見るだけになる。
"""

import sys
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

class BIT2:

    def __init__(self, N):
        N += 1
        self.N = N
        self.data0 = [0] * N
        self.data1 = [0] * N

    def _add(self, data, k, x):
        k += 1
        while k < self.N:
            data[k] += x
            k += k & -k

    def _get(self, data, k):
        k += 1
        s = 0
        while k:
            s += data[k]
            k -= k & -k
        return s

    def add(self, l, r, x):
        self._add(self.data0, l, -x*(l-1))
        self._add(self.data0, r, x*(r-1))
        self._add(self.data1, l, x)
        self._add(self.data1, r, -x)

    def query(self, l, r):
        return self._get(self.data1, r-1) * (r-1) + self._get(self.data0, r-1) \
             - self._get(self.data1, l-1) * (l-1) - self._get(self.data0, l-1)

    def get(self, i):
        return self.query(i, i+1)
    
    def update(self, i, x):
        self.add(i, i+1, x - self.get(i))

MAX = 10**5 * 2
bit = BIT2(MAX+1)
Q = INT()
n = 0
ans = []
for _ in range(Q):
    lr, id = input().split()
    id = int(id)
    if lr == '?':
        idx = bit.get(id)
        ans.append(str(min(idx, n-idx-1)))
        continue
    if lr == 'L':
        bit.add(0, MAX+1, 1)
        bit.update(id, 0)
    else:
        bit.update(id, n)
    n += 1
print('\n'.join(ans))
