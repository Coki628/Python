# -*- coding: utf-8 -*-

import sys

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

s = input()
N = len(s)

# 文字列を各カードに分割
i = 1
cards = []
while i < N:
    if s[i] =='1':
        cards.append(s[i-1:i+2])
        i += 3
    else:
        cards.append(s[i-1:i+1])
        i += 2

# 当たりの組4つ
S = [set() for i in range(4)]
S[0] = {'S10', 'SJ', 'SQ', 'SK', 'SA'}
S[1] = {'H10', 'HJ', 'HQ', 'HK', 'HA'}
S[2] = {'D10', 'DJ', 'DQ', 'DK', 'DA'}
S[3] = {'C10', 'CJ', 'CQ', 'CK', 'CA'}

def check(cards):
    cnt = [0] * 4
    for i, card in enumerate(cards):
        for j in range(4):
            if card in S[j]:
                cnt[j] += 1
                # いずれかの種類で5枚揃ったら終了位置と種類を返す
                if cnt[j] == 5:
                    return i, j
idx, num = check(cards)

# 終了位置までで、揃った5枚ではないものを回収する
ans = []
for i, card in enumerate(cards):
    if i == idx:
        break
    if card not in S[num]:
        ans.append(card)
if ans:
    print(''.join(ans))
else:
    print(0)
