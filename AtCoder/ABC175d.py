"""
参考：https://twitter.com/Euglenese/status/1294630479019175942
　　　https://atcoder.jp/contests/abc175/submissions/15949520
・これはすごい。こんなきれいにできるんだな。
・ダブリング応用
・やろうとしてたのに近い方針。自分ではK以下でのmaxは出せなかったけど、こうやる。
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
MOD = 10 ** 19 + 7
EPS = 10 ** -10

N, K = MAP()
A = [a-1 for a in LIST()]
C = LIST()
MAX = 32

N = len(A)
# nxt[k][i] := iから2^k回移動した後のマス
nxt = list2d(MAX, N, -1)
# score[k][i] := iから2^k回移動した時のスコア
score = list2d(MAX, N, -1)
# mx[k][i] := iから2^k回以下移動した時のスコアの最大値
mx = list2d(MAX, N, -1)
for i, a in enumerate(A):
    nxt[0][i] = a
    score[0][i] = C[a]
    mx[0][i] = C[a]

for k in range(1, MAX):
    for i in range(N):
        nxt[k][i] = nxt[k-1][nxt[k-1][i]]
        score[k][i] = score[k-1][i] + score[k-1][nxt[k-1][i]]
        # この遷移が分からなかった
        mx[k][i] = max(mx[k-1][i], score[k-1][i] + mx[k-1][nxt[k-1][i]])

ans = -INF
for i in range(N):
    cur = i
    sm = 0
    for k in range(MAX-1, -1, -1):
        if K >> k & 1:
            # ダブリングの遷移に従ってここでもよしなにマージする
            ans = max(ans, sm + mx[k][cur])
            sm += score[k][cur]
            cur = nxt[k][cur]
    ans = max(ans, sm)
print(ans)
