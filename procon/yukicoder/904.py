"""
・なんとか自力AC、試行錯誤長かった。。
・数え上げ
・方針決まってからWAがなかなか取れず。最終的に、駅1以外の入出数の割り振りによっては、
　駅1以外から入る人全部を駅1以外から出すのが(合計が足りてても)できないってことを見落としてると分かった。
　それを加味して条件整えたらなんとかAC。制約小さいのもあって、1ずつチェックして数え上げる方でもAC。
　一応、式変形して、数えるとこをO(1)にした方も無事AC。
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

N = INT()
A = [0] * (N-1)
B = [0] * (N-1)
for i in  range(N-1):
    A[i], B[i] = MAP()

smin = sum(A)
smout = sum(B)
# 駅1以外から入って駅1以外から出れる最大を取っておく
mxout = 0
for i, (a, b) in enumerate(zip(A, B)):
    mxout += min(a, smout - b)

# 駅1から入る人数をiとして固定して、条件を満たせるか確認(これでもAC)
# ans = 0
# i = 0
# while 1:
#     cnt = smin + i - smout
#     if 0 <= cnt and smout - i <= mxout:
#         if cnt <= smin:
#             ans += 1
#         else:
#             break
#     i += 1

# 上記数え上げの条件を式変形して、iの下限と上限をO(1)で出す
mn = max(smout - smin, smout - mxout, 0)
mx = smout
ans = max(mx - mn + 1, 0)
print(ans)
