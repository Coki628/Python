# -*- coding: utf-8 -*-

"""
・蟻本演習で飛ばしたやつ、無事ACできた。
・ランレングス圧縮応用
・実装おつ。うまくまとめられてよかった。
・カードの束をうまく区切る所が作れれば、後はだいたい大丈夫。
・カードの束は最初1つだけど、最大5000回かけて増えていくので、計算量は5000^2扱いだと思う。
・pythonAC5.8秒、pypyAC0.5秒。(制約10秒)
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

def shuffle(cards, x, y):
    cnt = 0
    i = 0
    A, B, C = [], [], []
    # 山Aを作る
    while cnt + (cards[i][1] - cards[i][0]) <= x:
        cnt += cards[i][1] - cards[i][0]
        A.append(cards[i])
        i += 1
    rest = x - cnt
    l, r = cards[i]
    cnt += rest
    if rest > 0:
        A.append((l, l+rest))
    cards[i] = (l+rest, r)
    # 山Bを作る
    while cnt + (cards[i][1] - cards[i][0]) <= y:
        cnt += cards[i][1] - cards[i][0]
        B.append(cards[i])
        i += 1
    rest = y - cnt
    l, r = cards[i]
    cnt += rest
    if rest > 0:
        B.append((l, l+rest))
    cards[i] = (l+rest, r)
    # 山Cを作る
    while i < len(cards):
        C.append(cards[i])
        i += 1
    # C,B,Aの並びにして返却
    return C + B + A

def check(cards, x, y, z):
    x -= 1
    cnt = 0
    i = 0
    res = []
    # やることは山Bを作るのとだいたい同じ
    while cnt + (cards[i][1] - cards[i][0]) <= x:
        cnt += cards[i][1] - cards[i][0]
        i += 1
    rest = x - cnt
    l, r = cards[i]
    cnt += rest
    cards[i] = (l+rest, r)
    while cnt + (cards[i][1] - cards[i][0]) <= y:
        cnt += cards[i][1] - cards[i][0]
        res.append(cards[i])
        i += 1
    rest = y - cnt
    l, r = cards[i]
    cnt += rest
    if rest > 0:
        res.append((l, l+rest))
    # 範囲内にあった束から、答えに該当する部分を数える
    ans = 0
    for l, r in res:
        ans += max(min(r, z+1) - l, 0)
    return ans

N = INT()
M = INT()
p, q, r = MAP()
cards  =[(1, N+1)]
for i in range(M):
    a, b = MAP()
    cards = shuffle(cards, a, b)

print(check(cards, p, q, r))
