"""
参考：https://twitter.com/mugen_1337/status/1274384149390831616
　　　https://codeforces.com/contest/1370/submission/84483099
・別解
・BITにぶたん
・set的なもので残っている0,1の位置を管理して、前から交互に取れるだけ取る、が操作1回。
　これを何回やるかで見る。
・こっちのが線形の解よりは直感的で思いつきやすいかも。
・でも100万にlog2つ乗るのはきつくて、pypyTLE。。
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
    """ Binary Indexed Tree """

    def __init__(self, n):
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

    def get(self, l, r=None):
        """ 区間和の取得 [l, r) """
        # 引数が1つなら一点の値を取得
        if r is None: r = l + 1
        res = 0
        if r: res += self.sum(r-1)
        if l: res -= self.sum(l-1)
        return res

    def update(self, i, x):
        """ 値の更新：添字i, 値x """
        self.add(i, x - self.get(i))

    def print(self, N):
        for i in range(N):
            print(self.get(i), end=' ')
        print()

    def bisearch_fore(self, l, r, x):
        """ 区間[l, r]を左から右に向かってx番目の値がある位置 """
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
            return INF

N = INT()
S = input()
T = input()

if S.count('0') != T.count('0'):
    print(-1)
    exit()

S2 = []
for i in range(N):
    if S[i] != T[i]:
        S2.append(S[i])
N2 = len(S2)

bit0 = BIT(N2)
bit1 = BIT(N2)
for i in range(N2):
    if S2[i] == '0':
        bit0.add(i, 1)
    else:
        bit1.add(i, 1)

ans = 0
while 1:
    idx0 = bit0.bisearch_fore(0, N2-1, 1)
    idx1 = bit1.bisearch_fore(0, N2-1, 1)
    # 全要素なくなったら終了
    if idx0 == INF:
        break
    # 1のが前にあったら1,0の順で取れるようにスワップ
    if idx0 > idx1:
        idx0, idx1 = idx1, idx0
        bit0, bit1 = bit1, bit0
    # 前から貪欲に残っている要素を交互に取っていく
    idx = idx0
    while 1:
        idx0 = bit0.bisearch_fore(idx, N2-1, 1)
        if idx0 == INF:
            break
        idx1 = bit1.bisearch_fore(idx0, N2-1, 1)
        if idx1 == INF:
            break
        bit0.add(idx0, -1)
        bit1.add(idx1, -1)
        idx = idx1
    # ここまでを1回の操作でできる
    ans += 1
print(ans)
