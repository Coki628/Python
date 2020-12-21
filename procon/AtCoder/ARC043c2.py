# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc043
・転倒数数え上げ応用、バブルソート
・案外、前半部分は自分の方針でほぼ合ってた。以下、後半部分について。
・転倒数0から、所定の転倒数の数列を作るのではなく、
　今バラバラのものから、所定の転倒数まで戻して(ソートして)いくことで、
　ソート済とバラバラの中間に当たるものが作れる。
・ちょうどいい位置まで戻すやり方としては、転倒数数え上げの情報から、近い所までまとめてやって、
　残りの部分はバブルソートを実際にシミュレーションして1つずつ数える。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

class BIT:
    """ Binary Indexed Tree """

    def __init__(self, n):
        # 0-indexed
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

N = INT()
A = [a-1 for a in LIST()]
B = [b-1 for b in LIST()]

atoi = {}
itoa = {}
for i, a in enumerate(A):
    atoi[a] = i
    itoa[i] = a

# 数列Aを順番通り(転倒数0)とみなした時のB
B2 = [atoi[b] for b in B]

# この転倒数がA,B間の転倒距離
bit = BIT(N)
cnt = 0
revs = [0] * N
for i in range(N):
    # 自分より左にある、自分より大きな数の個数
    tmp = i - bit.sum(B2[i])
    cnt += tmp
    # 各値毎の転倒数を取っておく
    revs[B2[i]] = tmp
    bit.add(B2[i], 1)

# ちょうど真ん中を作れない
if cnt % 2 == 1:
    print(-1)
    exit()

# 数列B2からtarget回分、転倒数を減らす
target = cnt // 2
cnt = j = 0
for i in range(N):
    # 各値個別の転倒数を取ってあるので、targetを超えない範囲までまとめて足す
    if cnt + revs[i] <= target:
        cnt += revs[i]
    else:
        j = i
        break
# ソート済部分
C = list(range(j))
# 未ソート部分
for b in B2:
    if b >= j:
        C.append(b)

# バブルソートを途中までシミュレーションして転倒数をtargetぴったりに合わせる
def bubble_sort(C, target):
    global cnt
    done = False
    end = 0
    while not done:
        done = True
        for i in range(N-1, end, -1):
            if C[i] < C[i-1]:
                C[i-1], C[i] = C[i], C[i-1]
                done = False
                cnt += 1
                # 所定の回数に達したらそこで打ち切る
                if cnt == target:
                    return C
        end += 1
    return C

if cnt < target:
    C = bubble_sort(C, target)

# 元の値に戻す
C2 = [itoa[c]+1 for c in C]
print(*C2)
