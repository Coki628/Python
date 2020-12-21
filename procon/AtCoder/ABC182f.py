"""
参考：https://atcoder.jp/contests/abc182/editorial/293
　　　https://atcoder.jp/contests/abc182/submissions/17987927
・題意から行われる操作を整理すると、大きい方から見た時、単位A[i]による支払いは
　-A[i] < x < A[i]の範囲内に収まる必要があり(そうでないと次の単位で収めきれない)、
　しかもA[i]ずつしか払えないので、ここには最大2通りしか場合がない。
　0ぴったりか、正の数と負の数に1つずつ。
　で、正の数or0ぴったりはA[i]でmodとればよく、負の数はそこから-A[i]すればいい。
・あと題意より、ひとつ上の単位以上に支払うことができないため、
　ここで出した額が大きすぎる場合は遷移させない。
・計算前の収支が負数だった時がちょっと違って、そのままA[i]でmod取ると意図した挙動にならないので、
　いったん正の数にしてからキーに収める時に符号を戻したらうまくいった。
・解説動画は桁DPっぽく管理して、実際の差額の値でなく、その単位をいくつ使ったかで管理してたっぽいけど、
　今回はその方針のがややこしそうに感じたので、解説PDFの方針で進めた。
・いずれにしろ、この考察を自力導出して解くのはまだまだ大変そうだ。。。
"""

import sys
from collections import defaultdict

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

N, X = MAP()
A  = LIST()

# 大きい単位から見る
A.reverse()
# dp[i][j] := i番目まで見て、Xとの差額がjの時の通り数
dp = [defaultdict(int) for i in range(N)]
# 少なめorぴったりに払う
x = X % A[0]
dp[0][x] += 1
# 多めに払う
x = X % A[0] - A[0]
if abs(x) < A[0]:
    dp[0][x] += 1
for i in range(1, N):
    for k, v in dp[i-1].items():
        # modの挙動を考慮して負数は場合分け
        if k >= 0:
            # 少なめorぴったりに払う
            x = k % A[i]
            # この単位で払う額が前の単位より小さいことが条件
            if abs(x-k) < A[i-1]:
                dp[i][x] += v
            # 多めに払う
            x = k % A[i] - A[i]
            if abs(x-k) < A[i-1]:
                dp[i][x] += v
        else:
            # mod取る前に正の数にする
            k = abs(k)
            x = k % A[i]
            if abs(x-k) < A[i-1]:
                # キーにする時、正負を戻す
                dp[i][-x] += v
            x = k % A[i] - A[i]
            if abs(x-k) < A[i-1]:
                dp[i][-x] += v
# 全部終わって収支が合っている通り数
ans = dp[-1][0]
print(ans)
