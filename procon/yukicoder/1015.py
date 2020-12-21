"""
参考：https://yukicoder.me/problems/no/1015/editorial
　　　https://www.hamayanhamayan.com/entry/2020/04/04/171929
・自力ならず。全然いい方針出ず。。
・高度貪欲考察、優先度付きキュー
・コンテスト中ならもうちょい粘って嘘貪欲並べてWA叩きまくったぽいけど、正解の方針に辿り着けた気は全くしない。。
"""

import sys
from heapq import heappush, heappop, heapify

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

N, X, Y, Z = MAP()
# 最初からa+1円必要と考える
que = [-(a+1) for a in LIST()]

heapify(que)
# 高い紙幣から貪欲に使う
for cur, cnt in [(10000, Z), (5000, Y), (1000, X)]:
    # この紙幣より高い金額をできるだけ減らす
    while que and cnt:
        a = -heappop(que)
        if a < cur:
            heappush(que, -a)
            break
        d, m = divmod(a, cur)
        if cnt >= d:
            cnt -= d
        else:
            d -= cnt
            cnt = 0
            m += d * cur
        if m:
            heappush(que, -m)
    # 安いのしかなくなったら、高い方から消化する
    while que and cnt:
        heappop(que)
        cnt -= 1
if not que:
    Yes()
else:
    No()
