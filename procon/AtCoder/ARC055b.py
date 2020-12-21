# -*- coding: utf-8 -*-

"""
参考：http://arc055.contest.atcoder.jp/data/arc/055/editorial.pdf
　　　http://mayokoex.hatenablog.com/entry/2016/06/04/102854
　　　http://baitop.hatenadiary.jp/entry/2018/09/11/015125
・確率系
・メモ化再帰
・添字的には、2次元DP+フラグの3次元
・今までで最大のものが出る確率は、実はNの大きさに関わらず1/i、
　これを戻り値に適用するので、1つ先に当たる1/(i+1)をかける。
・O(NK)=1000*1000=100万でpythonTLE、pypyで1.2秒AC。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, K = MAP()

# memo[i][j][is_max] := i枚目まで考えて、j枚食べている時、その時点で最大のものを取っているかどうか(is_max)、
#                       の状態での、全体で最大のものを取れる確率
memo = list3d(N+1, K+1, 2, -1)
def rec(i, j, is_max):
    if j > K:
        return 0
    # N枚終了時に、最大の物を取っているかどうか
    if i == N:
        return is_max
    if memo[i][j][is_max] != -1:
        return memo[i][j][is_max]
    res = 0
    # 次のiが最大値ではない時
    res += rec(i+1, j, is_max) * (i / (i+1))
    # 次のiが最大値の時
    res += max(
        # 取る
        rec(i+1, j+1, 1),
        # 取らない
        rec(i+1, j, 0),
    ) * (1 / (i+1))

    memo[i][j][is_max] = res
    return res

print(rec(0, 0, 0))
