"""
・自力AC！
・bitに状態を持たせるDP、円環
・円環なので開始状態を覚えておいて、終了状態との整合性をチェックする。
・あとは直前3つの状態を持ちながらDPを進めればいい。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N = INT()
B = LIST(N)

# 000,100,111は0
is_0 = set([0, 4, 7])
# dp[bit1][bit2][i] := 開始時の末尾と先頭ビットの状態がbit1で、直前3ビットの状態がbit2で、操作後をiまで見た時の通り数
dp = list3d(1<<2, 1<<3, N, 0)
for bit in range(1<<3):
    # 操作後の状態Bと矛盾がなければ初期値に追加
    if (bit in is_0) ^ B[0]:
        dp[bit>>1][bit][0] = 1

msk = (1<<3) - 1
for i in range(1, N):
    for init in range(1<<2):
        for bit in range(1<<3):
            for k in range(2):
                nxt = ((bit<<1)&msk) | k
                # 操作後の状態Bと矛盾がなければ遷移させる
                if (nxt in is_0) ^ B[i]:
                    dp[init][nxt][i] += dp[init][bit][i-1]
                    dp[init][nxt][i] %= MOD

ans = 0
msk = (1<<2) - 1
for init in range(1<<2):
    for bit in range(1<<3):
        # 開始位置と終了位置で矛盾がないところだけ答えに足す
        if init == (bit&msk):
            ans += dp[init][bit][-1]
            ans %= MOD
print(ans)
