"""
参考：https://twitter.com/SSRS_cp/status/1321128615111655426
・自力ならず。。このDP解けないのは悔しすぎる。。
・ちょっとド典型じゃない2次元DP、ソートしてDP
・ずっとi個目を取り出した時に時間jだったら〜の向きで考えてしまっていたんだけど、
　時間iまで見てj個取り出した時の〜の向きにするとうまくいくやつだった。。。
・普通にナップザックっぽく考えると、取り出す料理をiでインデックスにしたくなるけど、
　ここは逆にするんだよねー。でもたまにあるよねこういうの。
・なんとなく3乗間に合う制約なのも騙されたなー。もっと複雑に考えてしまった。
　結局遷移2つだけだから普通の2乗DPじゃん。。
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
    A = LIST()

    A.sort()
    dp = list2d(N*2+1, N+1, INF)
    dp[0][0] = 0
    for i in range(N*2):
        for j in range(N+1):
            # 時間i+1には何も取り出さない
            dp[i+1][j] = min(dp[i+1][j], dp[i][j])
            if j+1 <= N:
                # 時間i+1で料理jを取り出す
                dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j] + abs((i+1) - A[j]))
    ans = INF
    for i in range(N*2+1):
        ans = min(ans, dp[i][N])
    print(ans)
