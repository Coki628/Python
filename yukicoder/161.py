"""
・自力AC
・頭いい貪欲ありそうだなーと思いつつも制約見てDPに。(想定解はやっぱ貪欲だった)
・まともにやると最大300*100^3=3億できつい。C++に走ろうと思ったところで次元を落とせると気付いた。
・全体の終わった回数-グー回数-チョキ回数ってやればパーの回数が出る。
　これでループ1つ削って300*150^2=700万弱、pypyAC0.5秒。
・ちなみにl<0のスキップ処理はcontinueでもよかったけどbreakにした。考えたらjとkは単調増加だもんね。
　枝刈りになるかと思ったけど、最大ケースでは変わらないみたいで全体としては時間は変化なかった。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

g, c, p = MAP()
S = input()
N = len(S)

def calc(a, b):
    if a == b:
        return 1
    if a == 'G' and b == 'C' or a == 'C' and b == 'P' or a == 'P' and b == 'G':
        return 3
    if a == 'G' and b == 'P' or a == 'C' and b == 'G' or a == 'P' and b == 'C':
        return 0

dp = list2d(g+1, c+1, -INF)
nxt = list2d(g+1, c+1, -INF)
dp[0][0] = 0
for i, s in enumerate(S):
    for j in range(g+1):
        for k in range(c+1):
            # パーを出した回数l
            l = i - j - k
            # ありえない値になったらスキップ
            if l < 0:
                break
            if j+1 <= g:
                nxt[j+1][k] = max(nxt[j+1][k], dp[j][k] + calc('G', s))
            if k+1 <= c:
                nxt[j][k+1] = max(nxt[j][k+1], dp[j][k] + calc('C', s))
            if l+1 <= p:
                nxt[j][k] = max(nxt[j][k], dp[j][k] + calc('P', s))
    dp = nxt
    nxt = list2d(g+1, c+1, -INF)
ans = dp[g][c]
print(ans)
