"""
・自力AC
・ゲーム系、偶奇、貪欲
・それぞれが取れる場所が決まっているので、相手の行動にそこは影響されない。
　ので、自分が残すと負ける方から貪欲に減らしていく。で、最後に残ったやつで判定。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N = INT()
    S = list(map(int, input()))

    cnt = list2d(2, 2, 0)
    for i in range(N):
        cnt[i&1][S[i]&1] += 1
    for i in range(N-1):
        if i % 2 == 0:
            if cnt[i&1][0]:
                cnt[i&1][0] -= 1
            else:
                cnt[i&1][1] -= 1
        else:
            if cnt[i&1][1]:
                cnt[i&1][1] -= 1
            else:
                cnt[i&1][0] -= 1
    if cnt[0][1] or cnt[1][1]:
        print(1)
    else:
        print(2)
