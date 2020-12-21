"""
・終了後、システス待ってる間で書いて、システス終わってから出してWAだったけどちょっと直してすぐAC。。
・2種類貪欲する。?の部分は1111....0000と0000...1111のどっちかが最適。
　先に全0か全1で計算してそこからスタートする。
　?を1つ切り替えた時に変化するのはその位置が関係する4通りだけ。
　この全状態で最小を取ればOK。01の個数はBIT使えばうまく動的に変化させられる。
・ちゃんと考えたら結構すんなり考察進んで通せたなー。
　Cなんとか通ったから良かったけど、あんな詰まるならこっち行ってもよかったな。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

class BIT:

    def __init__(self, n):
        self.n = n
        n += 1
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [0] * nv

    def sum(self, i):
        s = 0
        i += 1
        while i > 0:
            s += self.tree[i-1]
            i -= i & -i
        return s

    def add(self, i, x):
        i += 1
        while i <= self.size:
            self.tree[i-1] += x
            i += i & -i

    def query(self, l, r):
        return self.sum(r-1) - self.sum(l-1)
    
    def get(self, i):
        return self.query(i, i+1)

    def update(self, i, x):
        self.add(i, x - self.get(i))

    def print(self):
        for i in range(self.n):
            print(self.get(i), end=' ')
        print()

S = list(input())
N = len(S)
x, y = MAP()

bit = BIT(N)
for i in range(N):
    if S[i] == '1':
        bit.add(i, 1)
cnt = 0
for i in range(N):
    if bit.get(i) == 0:
        cnt += bit.query(i+1, N)*x
    else:
        cnt += ((N-i-1)-bit.query(i+1, N))*y
ans = cnt
for i in range(N):
    # 0 -> 1
    if S[i] == '?':
        # 前の01が増える
        cnt += (i-bit.query(0, i))*x
        # 前の10が減る
        cnt -= bit.query(0, i)*y
        # 後ろの01が減る
        cnt -= bit.query(i+1, N)*x
        # 後ろの10が増える
        cnt += ((N-i-1)-bit.query(i+1, N))*y
        bit.add(i, 1)
    ans = min(ans, cnt)

bit = BIT(N)
for i in range(N):
    if S[i] == '1' or S[i] == '?':
        bit.add(i, 1)
cnt = 0
for i in range(N):
    if bit.get(i) == 0:
        cnt += bit.query(i+1, N)*x
    else:
        cnt += ((N-i-1)-bit.query(i+1, N))*y
ans = min(ans, cnt)
for i in range(N):
    # 1 -> 0
    if S[i] == '?':
        # 前の01が減る
        cnt -= (i-bit.query(0, i))*x
        # 前の10が増える
        cnt += bit.query(0, i)*y
        # 後ろの01が増える
        cnt += bit.query(i+1, N)*x
        # 後ろの10が減る
        cnt -= ((N-i-1)-bit.query(i+1, N))*y
        bit.add(i, -1)
    ans = min(ans, cnt)

print(ans)
