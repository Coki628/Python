# -*- coding: utf-8 -*-

"""
参考：http://abc038.contest.atcoder.jp/data/abc/038/editorial.pdf
・2つのDP(1次元DPとBIT)
・BITはここでは区間和ではなく最大値を取る。
・しかしこの添え字の当て方はまだ自分には出せなかったな。
　データの持たせ方大事。
"""

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7
 
class BIT:
 
    def __init__(self, n, init):
        # 0-indexed
        nv = 1
        while nv < n:
            nv *= 2
        self.size = nv
        self.tree = [init] * nv
 
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
 
    def max(self, i):
        """ [0, i]の最大値 """
        s = -INF
        i += 1
        while i > 0:
            s = max(s, self.tree[i-1])
            i -= i & -i
        return s
 
    def update_max(self, i, x):
        """ 最大値の更新：添字i, 値x """
        i += 1
        while i <= self.size:
            self.tree[i-1] = max(self.tree[i-1], x)
            i += i & -i
 
N=INT()
HW=[]
for i in range(N):
    w,h=MAP()
    # 紛らわしいので縦横の順にしておく
    HW.append((h, w))
 
# 縦の昇順、横の降順に並べておく(横に降順の理由は公式解説と自分のノート参照)
HW.sort(key=itemgetter(1), reverse=True)
HW.sort(key=itemgetter(0))
 
# dp[i] := i番目の箱を最も外側とするとき、最大で何重の入れ子とできるか
dp=[0]*N
# bit[i] := 最も外側の箱の横の長さがiとなる入れ子の数の最大値
bit=BIT(100001, 0)
 
# 縦が小さい方から決めていく
for i in range(N):
    h,w=HW[i]
    # 横が自分より小さいものの中から最大を取る(+自分の分で1)
    dp[i]=bit.max(w-1)+1
    # 今回分は決まったので、それに合わせてBITも更新
    bit.update_max(w, dp[i])
 
print(max(dp))
