"""
・終了後約5分で自力AC、、悔しいー。。
・累積和
・この問題ではある区間が遮られて、その残った前後の部分を足す。
　この前後部分をなんとか累積和とかでまとめられないか考える。
・最初、普通に左端固定で右端位置をまとめる方針を取ろうとしたら、前部分で桁位置が変わってまとめられず詰む。
・長らく色々悩んだ後、左端じゃなく右端を固定すれば前部分の桁位置が一定になると分かる。
　(この時終了1分前くらい。。。)
・で、その通りに前計算の累積和取って投げたら無事AC。これバチャじゃなかったら相当悔しかったなー。。
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

S = input()
N = len(S)

# suf[i] := i~Nを連結させた値(を後から累積和する)
suf = [0] * (N+1)
for i in range(N-1, -1, -1):
    suf[i] = (suf[i+1] + int(S[i])*pow(10, N-i-1, MOD)) % MOD
for i in range(N-1, -1, -1):
    suf[i] += suf[i+1]
    suf[i] %= MOD

# pre[i] := 0~iを連結させた値(を前から累積和する)
pre = [0] * (N+1)
for i in range(N-1):
    pre[i+1] = (pre[i]*10 + int(S[i])) % MOD
for i in range(N):
    pre[i+1] += pre[i]
    pre[i+1] %= MOD

ans = 0
for i in range(N-1, -1, -1):
    # 後ろの桁数分、前の桁をずらして足す
    ans += pre[i]*pow(10, N-i-1, MOD) + suf[i+1]
    ans %= MOD
print(ans)
