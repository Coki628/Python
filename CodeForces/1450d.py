"""
・なんとか自力AC！
・BITにぶたん
・順列を構成しないといけないので、1から順にその値が存在できる場所が1箇所あるかを考える。
　ある値aを新しく答えに登場させるためには、その時の答えが取りうる区間長で、
　ちょうど1回その値aが最小値になる必要がある。
　それが1つもなかったり2つ以上あったらその区間長はNG。
・それとは別に、値aが1つも元の配列に含まれていなかったら、その先は全部NG。
・最初の条件の方で、最初continueしてたらWAで、実はこっちも2つめの条件と一緒でその先全部NGじゃないか？
　となって、でもサンプルからしても区間長1の時は全部揃ってれば絶対OKだから、
　そこだけ例外処理して、件のcontinueはbreakにして、とりあえず投げちゃえ、えいってやったらAC。。。
・粘って通せたのは嬉しいけど、こういう雰囲気ACみたいなのは嬉しくないよなー。。
・30万のNにlog2つはこどふぉ環境だとpypyきついかもって思ったけど、さすがBITだと軽くて、なんとかAC1.25秒。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class BIT:

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

    def bisearch_fore(self, l, r, x):
        l_sm = self.sum(l-1)
        ok = r + 1
        ng = l - 1
        while ng+1 < ok:
            mid = (ok+ng) // 2
            if self.sum(mid) - l_sm >= x:
                ok = mid
            else:
                ng = mid
        if ok != r + 1:
            return ok
        else:
            return -1

    def bisearch_back(self, l, r, x):
        r_sm = self.sum(r)
        ok = l - 1
        ng = r + 1
        while ok+1 < ng:
            mid = (ok+ng) // 2
            if r_sm - self.sum(mid-1) >= x:
                ok = mid
            else:
                ng = mid
        if ok != l - 1:
            return ok
        else:
            return -1

for _ in range(INT()):
    N = INT()
    A = [a-1 for a in LIST()]
    C = [0] * N
    B = [[] for i in range(N)]
    for i in range(N):
        C[A[i]] += 1
        B[A[i]].append(i)

    bit = BIT(N)
    ans = [0] * N
    for a in range(N):
        if not C[a]:
            break
        ln = N - a - 1
        se = set()
        for i in B[a]:
            l = bit.bisearch_back(max(i-ln, 0), i-1, 1)
            if l == -1:
                l = max(i-ln, 0)
            else:
                l += 1
            r = bit.bisearch_fore(i+1, min(i+ln, N-1), 1)
            if r == -1:
                r = min(i+ln, N-1)
            else:
                r -= 1
            if r-l == ln:
                se.add((l, r))
            bit.add(i, 1)
        if len(se) != 1:
            break
        ans[ln] = 1
    if len(set(A)) == N:
        ans[0] = 1
    print(''.join(map(str, (ans))))
