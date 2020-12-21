"""
参考：https://atcoder.jp/contests/hhkb2020/editorial/178
・自力ならず、、体調万全だったら頭回ったかなぁ。。
・グリッド、累積和、総和の数え上げ、言い換え
・各マスについて、「そのマスが照らされる回数」を出せれば、全マス見た時に題意のものが得られる。
・公式解は全体から引く方針にしてたけど、素直にやってもいけた。
　そのマスを照らせないパターンは同じグループが全てランプなしになる状況だけなので、
　照らされるのは、同じグループ内での置き方の全通り-1、これに関係ない場所全通りを掛ける。
・最初普通にやったらTLEで、400万グリッドくらいなら通るはずなのになー、と思った。
　本番中なら気合いでC++書き直しとかやってたかもしれないけど、とりあえず人様の成功例を確認。
　いくつか見て、グリッド1次元にしてるのもあったけど、めんどいしそれはやらなくても
　なんとかならんかなーって思って、modpowのlogを疑ったら当たり。
　400万のループ内で毎回modpowしてるのがきつかったらしく、ここを前計算したら無事pypyAC0.73秒。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

H, W = MAP()
grid = []
dotcnt = 0
for i in range(H):
    row = input()
    grid.append(row)
    dotcnt += row.count('.')

accl = list2d(H, W, 0)
accr = list2d(H, W, 0)
for i in range(H):
    for j in range(W-1):
        if grid[i][j] == '.':
            accl[i][j+1] = accl[i][j] + 1
    for j in range(W-1, 0, -1):
        if grid[i][j] == '.':
            accr[i][j-1] = accr[i][j] + 1
accd = list2d(H, W, 0)
accu = list2d(H, W, 0)
for j in range(W):
    for i in range(H-1):
        if grid[i][j] == '.':
            accd[i+1][j] = accd[i][j] + 1
    for i in range(H-1, 0, -1):
        if grid[i][j] == '.':
            accu[i-1][j] = accu[i][j] + 1

# 2冪前計算(これやんないとTLE)
pow2 = [0] * (dotcnt+1)
cur = pow2[0] = 1
for i in range(1, dotcnt+1):
    cur *= 2
    cur %= MOD
    pow2[i] = cur

ans = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            cnt = accl[i][j] + accr[i][j] + accd[i][j] + accu[i][j] + 1
            ans += (pow2[cnt]-1) * pow2[dotcnt-cnt] % MOD
            ans %= MOD
print(ans)
