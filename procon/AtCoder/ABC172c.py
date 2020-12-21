"""
・最初貪欲やって死亡。冷静になってよくよく考える必要があった。
・累積和、二分探索
・C問題で累積和とにぶたんとか攻めてる。平成ABCみたい。
　でもこういう数学じゃなくて競プロ的な難易度の上げ方は好き。数学勢に勝てるからね。。
"""

import sys
from itertools import accumulate
from bisect import bisect_right

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

N, M, K = MAP()
A = LIST()
B = LIST()

acc1 = [0] + list(accumulate(A))
acc2 = [0] + list(accumulate(B))
ans = 0
for i in range(N+1):
    cur = K - acc1[i]
    if cur < 0:
        break
    j = bisect_right(acc2, cur) - 1
    ans = max(ans, i+j)
print(ans)
