# -*- coding: utf-8 -*-

"""
参考：https://wata-orz.hatenadiary.org/entry/20090922/1253615708
　　　https://atcoder.jp/contests/jag2013summer-warmingup/submissions/813937
・区間スケジューリング、右端でソート、ダブリング、尺取法
・各始点からの最初の1回の求め方もちょっと工夫が必要で、尺取っぽくlとrを持って、
　右端ソートした列に対して、rの左端がlの右端以上になるまで、rを進める。
・制約(1.5秒、96M)つらかったけど、何とかpypyAC0.5秒69M。
・2周しないとダメだと思ってて、N*2のテーブル作ってMLEしてたんだけど、実は1周分だけで大丈夫だった。
　周回する右端は最初に+Mしていたので、右端ソートでうまく後ろに行って、よしなに動いてくれてた。
"""

import sys
from operator import itemgetter

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

N, M = MAP()
ST = []
for i in range(N):
    s, t = MAP()
    if t < s:
        t += M
    ST.append((s, t))
# 右端でソート
ST.sort(key=itemgetter(1))

# nxt[k][i] := iからスタートして、2^k回移動した後にいる位置
nxt = list2d(17, N, -1)

# 尺取法
l = r = 0
while l < N:
    # 左端が自分の右端以上になるまでrを進める
    while r < N and ST[l][1] > ST[r][0]:
        r += 1
    if r < N:
        nxt[0][l] = r
    l += 1

# ダブリングのテーブル構築
for k in range(1, 17):
    for i in range(N):
        nxt[k][i] = nxt[k-1][nxt[k-1][i]]   

ans = 0
for i in range(N):
    s, _ = ST[i]
    cnt = cur = 0
    # 何回先まで行けるか
    for k in range(16, -1, -1):
        if nxt[k][i] != -1:
            _, t = ST[nxt[k][i]]
            # 区間長の合計がM以内になるような最大にする
            if cur + (t - s) <= M:
                cnt += 2 ** k
                cur += t - s
                s = t
                i = nxt[k][i]
    ans = max(ans, cnt)
# 自分の分を足す
print(ans + 1)
