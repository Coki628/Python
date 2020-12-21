"""
参考：https://yukicoder.me/problems/no/1043/editorial
　　　https://www.hamayanhamayan.com/entry/2020/05/02/002259
・自力ならず。
・部分和DP、累積和、式変形
・解法出なかったー。前も部分和DP出なかったんだよな、反省したい。
・制約がN^2いける程度には弱めで、指数時間は無理だけど和の取り方を全列挙しなきゃいけなそうっていう状況は、
　部分和DPを疑うべきだね。。
・今回、取りうる値が10万まであって、要素が100個あるから、ちょっときつそうに見えるけど1000万は全然間に合う。
・Rを全探索してVで区間を取る方にした。こっちのが式変形が掛け算で楽、、と思ったら落とし穴があって
　こっちのがむしろ最初WA出した。。
・式変形して電圧Vが値が大きくなりすぎたらcontinueしてたんだけど、
　最大値がオーバーしてるだけだからその区間が取れない訳じゃないね。。
　ってことでMXとminを取るようにして無事AC。
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

N, M = MAP()
V = LIST()
R = LIST()
L, H = MAP()
MX = 100000

# 電圧Vの部分和とその通り数を全列挙
dp1 = list2d(2, MX+1, 0)
dp1[0][0] = 1
for i, v in enumerate(V):
    for j in range(MX+1):
        if dp1[i&1][j]:
            dp1[(i+1)&1][j] += dp1[i&1][j]
            dp1[(i+1)&1][j+v] += dp1[i&1][j]
            dp1[(i+1)&1][j] %= MOD
            dp1[(i+1)&1][j+v] %= MOD
            dp1[i&1][j] = 0

# 抵抗Rの部分和とその通り数を全列挙
dp2 = list2d(2, MX+1, 0)
dp2[0][0] = 1
for i, r in enumerate(R):
    for j in range(MX+1):
        if dp2[i&1][j]:
            dp2[(i+1)&1][j] += dp2[i&1][j]
            dp2[(i+1)&1][j+r] += dp2[i&1][j]
            dp2[(i+1)&1][j] %= MOD
            dp2[(i+1)&1][j+r] %= MOD
            dp2[i&1][j] = 0

# 電圧の部分和を累積和して区間を取れるようにする
acc = list(accumulate(dp1[N&1]))
ans = 0
# 抵抗rを全探索して取れる電圧の範囲で区間を足していく
for r, cnt in enumerate(dp2[M&1]):
    if r == 0:
        continue
    lowv = min(L * r, MX)
    hiv = min(H * r, MX)
    ans += cnt * (acc[hiv] - acc[lowv-1])
    ans %= MOD
print(ans)
