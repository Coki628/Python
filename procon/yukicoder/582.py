"""
・なんとか自力AC
・ゲーム、場合分け
・1つ山についてゲームDPしてみると、Aは自分のターンでラスト1個の状況以外では勝てないと分かる。
・複数山を考えてみると、ラス1の山のみ残ってかつその山数が偶数個、の状況をBに渡せた場合のみ勝てる、となる。
・なんかWAしてしばらく悩んで、Aの初手のみで決着がつくケースを考慮できてなかったと判明。
・1つ山の勝敗とか、このくらいゲームDPしなくても考察できたいよなぁ。
・なんとなく複数山だしNimとかGrundyみたいな香りが漂うんだけど、両者のできる操作が違うので、
　この場合はGrundyでXOR、みたいなやつは適用できないと思う。
"""

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
EPS = 10 ** -10

memo = list2d(2, 30, -1)
def rec(turn, cur):
    if memo[turn][cur] != -1:
        return memo[turn][cur]
    if cur == 0:
        return 0
    res = 1
    if turn % 2 == 0:
        res &= rec(1-turn, cur-1)
    else:
        res &= rec(1-turn, cur-1)
        if cur-2 >= 0:
            res &= rec(1-turn, cur-2)
    if not res:
        memo[turn][cur] = 1
        return 1
    else:
        memo[turn][cur] = 0
        return 0

# for i in range(1, 30):
#     rec(0, i)

N = INT()
C = LIST()

mxi = C.index(max(C))
C[mxi] -= 1

if sum(C) == 0 or max(C) == 1 and C.count(1) % 2 == 0:
    print('A')
else:
    print('B')
