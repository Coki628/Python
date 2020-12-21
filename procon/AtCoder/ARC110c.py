"""
・なんとか自力AC！こっちは結構早かった。
・順列、転倒数、構築
・まず、スワップ回数(転倒数)がN-1回以外だったらNG。
・そうでないとして、前から見て結局各数が希望のスワップ回数左に動くのをできなかったらNG。
　OKならその通りに左から順に動かせばいい。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class BIT:
    """ Binary Indexed Tree """

    def __init__(self, n):
        self.n = n
        # 0-indexed
        n += 1
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [0] * nv

    def sum(self, i):
        """ [0, i]を合計する """
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    def add(self, i, x):
        """ 値の追加：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

    def query(self, l, r):
        """ 区間和の取得 [l, r) """
        return self.sum(r-1) - self.sum(l-1)
    
    def get(self, i):
        """ 一点取得 """
        return self.query(i, i+1)

    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        self.add(i, x - self.get(i))

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

N = INT()
A = [a-1 for a in LIST()]

bit = BIT(N)
cnt = [0] * N
for i in range(N):
    cnt[i] = bit.query(A[i], N)
    bit.add(A[i], 1)

if sum(cnt) != N-1:
    print(-1)
    exit()

ans = []
used = [0] * N
for i in range(N):
    if cnt[i]:
        j = i
        c = 0
        while c < cnt[i]:
            c += 1
            j -= 1
            if used[j]:
                print(-1)
                exit()
            used[j] = 1
            ans.append(j+1)
for a in ans:
    print(a)
