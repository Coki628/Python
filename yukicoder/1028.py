"""
参考：https://drive.google.com/file/d/1KKJHppIV_zX0ds-AyJnAyqAAr0kiQDZZ/view
・自力ならず。。。
・いもす応用
・writerさん解はなんか難しそうで、testerさん解のが自分がやろうとしてた方向性に近かったのでこっちで。
・区間をうまくいもすで管理。距離によって増加減少する区間をどうまとめるかうまくできなかった。。
　これは、いもすした数列をもう1回累積和取ると、ちょうどある位置から見たコストを作れる。
・それが分かれば、あとは増加部分と減少部分で分けていもすを取って累積和しておいて、
　それとは別に固定で足される部分を加味すれば、ある位置iで社員aに対してするコストをO(1)で取れる。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N = INT()
imos1 = list2d(N, N+1, 0)
imos2 = list2d(N, N+1, 0)
fixed = [0] * N
ll = 0
rr = N
for i in range(N):
    for j, a in enumerate([a-1 for a in LIST()]):
        lr = max(i-j, 0)
        rl = min(i+j+1, N)
        imos1[a][ll] += 1
        imos1[a][lr] -= 1
        imos2[a][rl] += 1
        imos2[a][rr] -= 1
        fixed[a] += j
for i in range(N):
    imos1[i] = list(accumulate(imos1[i]))
    imos2[i] = list(accumulate(imos2[i]))
    imos1[i] = list(accumulate(imos1[i][::-1]))[::-1]
    imos2[i] = list(accumulate(imos2[i]))

ans = 0
for a in range(N):
    mn = INF
    for i in range(N):
        mn = min(mn, imos1[a][i] + imos2[a][i] + fixed[a])
    ans += mn
print(ans)
