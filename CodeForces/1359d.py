"""
参考：https://twitter.com/laycrs/status/1266045441205874688
・計算量的には問題ないはずなのにTLE。。
・色々調べたところ、10万の配列生成を60回やってるのがきついみたい。
　600万の数列1回作ることくらいは全然あるのにな。。
・累積和応用
・取りうる値が61種類しかないので、ある値を最大値とした時の区間和最大、をやる。
・ある値より大きい値を-INFにしておくことで、そういった値を含む区間を無効にできる。
・区間和の最大値は、前から累積minした値に左端を固定して、右端を動かしていけばいい。
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
INF = 10 ** 10
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

nums = sorted(set(A), reverse=1)
ans = -INF
for num in nums:
    for i in range(N):
        if A[i] > num:
            A[i] = -INF
    acc = [0] + list(accumulate(A))
    accmn = list(accumulate(acc, min))
    mx = -INF
    for i in range(N):
        mx = max(mx, acc[i+1]-accmn[i])
    ans = max(ans, mx-num)
print(ans)
