"""
・自力AC
・こどふぉDiv. 2のAとかですごい出そうな、丁寧に場合分けこねくり回すO(1)のやつ。
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

X, K, D = MAP()

# 最初に負数方向を考えなくていいと気づくとだいぶ楽になる
if X < 0:
    X = -X

d = X // D
if d >= K:
    # 足りないなら行けるだけ0に近づくだけ
    ans = X - D*K
else:
    # 足りるならまず0直前まで近づく
    ans = X - D*d
    K -= d
    # 偶数回残ってればそこに戻るし、奇数回なら0通り過ぎた反対側
    if K % 2 == 1:
        ans = -(ans - D)
print(ans)
