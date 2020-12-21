# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2019/02/27/000600
・ゲーム木の探索
・メモ化再帰
・後退解析：負け状態を押し付けられるかどうか
・ゲームDPするのは考えたんだけど、制約でかすぎて無理だなってなってた。
・今回は、小さいケースでゲームDPしてみて、規則性を見つけるって内容だった。
　やってみるべきだったな。。
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

N = INT()

memo = [-1] * 101
def rec(cur):
    if memo[cur] != -1:
        return memo[cur]
    # 残数が0になったら負け確定
    if cur == 0:
        memo[cur] = 0
        return 0
    nxt = 1
    res = 0
    while nxt <= cur:
        # 次ターンを負け状態に出来るなら、このターンは勝ち
        if not rec(cur-nxt):
            res = 1
        nxt *= 8
    # 自分が出来る手を全部試して、勝ち状態に出来なければここは負け状態で0のまま
    memo[cur] = res
    return res

# ans = []
# for i in range(101):
#     print('{0}: {1}'.format(i, rec(i)))
#     ans.append(rec(i))
# print(','.join(map(str, (ans))))

# 規則性(mod9で循環)が分かったら、その通りに結果出力
ans = N % 9
if ans in [1,3,5,7,8]:
    print('Win')
else:
    print('Lose')
