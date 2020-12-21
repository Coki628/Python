"""
参考：皆様のツイート
・自力ならず。。悔しいなー。方針立たなかった。
・効率よく全探索、ソートして貪欲、累積和
・キレられる境界で2つに分けて要素を保持。(これはやった)
・まず小さい方は使う数が自由に決められる。なので小さい方の値の数を決め打つと、
　大きい方が残り何回使えるかが一意に定まる。
・こうなれば後は両方とも大きい順に使っていけばいいだけ。
　各個数使った時の値を前計算で累積和しておけば、それぞれO(1)で全部済む。
・この貪欲考察むずいと思うんだよなー。よくて大きい方を使う数を決め打ってしまいそう。
　でも今になって考えると、多分そっち決め打つ方がだいぶめんどくさい。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

N, D, M = MAP()
A1 = []
A2 = []
for a in LIST():
    if a <= M:
        A1.append(a)
    else:
        A2.append(a) 

A1.sort(reverse=1)
A2.sort(reverse=1)
N1 = len(A1)
N2 = len(A2)
acc1 = [0] + list(accumulate(A1))
acc2 = [0] + list(accumulate(A2))

ans = 0
for i in range(N1+1):
    have = N - i
    j = ceil(have, D+1)
    if j > N2:
        continue
    ans = max(ans, acc1[i] + acc2[j])
print(ans)
