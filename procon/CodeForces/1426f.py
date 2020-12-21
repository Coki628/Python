"""
・Dをなんとか通した後、残り時間も30分そこそこしかない中、フレンドの状況を見ると
　意外とEよりFを通してる人がいそうな雰囲気だったので、
　Eをぱっと見て疑わしい貪欲しか浮かばなかったのもあり、こっちを先に着手。
　結果的にはこれが功を奏して流れを引き寄せた感はある。
・耳DP
・状態を4つ、部分列開始前、aまで使った、bまで使った、cまで使った、を持ってDPする。
・?のところはa,b,cどれでも使えて遷移がたくさん増えるので、その通りにやる。
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

N = INT()
S = input()

dp0 = [0] * (N+1)
dp1 = [0] * (N+1)
dp2 = [0] * (N+1)
dp3 = [0] * (N+1)
dp0[0] = 1
for i in range(N):
    if S[i] == 'a':
        dp0[i+1] += dp0[i]
        dp1[i+1] += dp1[i]
        dp2[i+1] += dp2[i]
        dp3[i+1] += dp3[i]
        dp1[i+1] += dp0[i]
    elif S[i] == 'b':
        dp0[i+1] += dp0[i]
        dp1[i+1] += dp1[i]
        dp2[i+1] += dp2[i]
        dp3[i+1] += dp3[i]
        dp2[i+1] += dp1[i]
    elif S[i] == 'c':
        dp0[i+1] += dp0[i]
        dp1[i+1] += dp1[i]
        dp2[i+1] += dp2[i]
        dp3[i+1] += dp3[i]
        dp3[i+1] += dp2[i]
    else:
        dp0[i+1] += dp0[i]*3
        dp1[i+1] += dp1[i]*3
        dp2[i+1] += dp2[i]*3
        dp3[i+1] += dp3[i]*3
        dp1[i+1] += dp0[i]
        dp2[i+1] += dp1[i]
        dp3[i+1] += dp2[i]
    dp0[i+1] %= MOD
    dp1[i+1] %= MOD
    dp2[i+1] %= MOD
    dp3[i+1] %= MOD
ans = dp3[N]
print(ans)
