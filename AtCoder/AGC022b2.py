# -*- coding: utf-8 -*-

"""
参考：https://competitive12.blogspot.com/2019/09/agc-022-b-gcd-sequence.html
　　　https://img.atcoder.jp/agc022/editorial.pdf
・端折って言うと、総和がmod6で0になるように2,3の倍数を追加していけばOK。
・こういう整数系アドホックみたいなのほんとまだまだ無理だー。。
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

N = INT()

# 例外処理
if N == 3:
    print(2, 5, 63)
    exit()

# ・全体のgcdが1になるように2と3を含める
# ・要素は全て2か3の倍数を使う
# ・全体の合計は6の倍数にする
# ・これでgcd(総和, 各要素)は常に2か3になる
A = [2, 4, 3, 9]

li1, li2, li3 = [], [], []
for a in range(5, 30001):
    # 9は既に使用済
    if a == 9:
        continue
    # 3の倍数で奇数(mod6で3)
    if a % 3 == 0 and a % 2 == 1:
        li1.append(a)
    # 3の倍数で偶数(mod6で0)
    elif a % 6 == 0:
        li2.append(a)
    # 3の倍数ではなくて偶数(mod6で2,4)
    elif a % 3 != 0 and a % 2 == 0:
        li3.append(a)

# リスト1,3はペアで使うとmod6で0にできる
while li1 and len(A) + 2 <= N:
    A.append(li1.pop())
    A.append(li1.pop())
while li3 and len(A) + 2 <= N:
    A.append(li3.pop())
    A.append(li3.pop())
# リスト2は元々mod6で0なので1つずつ使える
while li2 and len(A) + 1 <= N:
    A.append(li2.pop())
print(*A)
