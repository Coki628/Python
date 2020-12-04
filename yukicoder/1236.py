"""
・なんとか自力AC。。
・時計の針
・時間を制御するやつはいつやっても疲れるね。。
・誤差で死にそうだったので、一番小さい秒単位まで細かくして12時間でmod取った。
・公式解は角度を使ってうまいことやるようなことが書いてあった。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

h, m = MAP()

# 時計1周を秒単位に区切る
mod = 12*60*60

# 短針の位置
hms = (h*60+m)*60 % mod
# 長針の位置
ms = 12*m*60 % mod
ans = 0
while 1:
    # print(hms, ms)
    # < 12 だと次でぴったり追いつくやつを拾ってしまうので < 11
    if abs(hms-ms) < 11:
        print(ans)
        break
    # 1秒分進める
    hms += 1
    hms %= mod
    ms += 12
    ms %= mod
    ans += 1
