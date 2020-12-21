"""
・自力AC
・木構造、完全二分木、再帰探索
・最初問題の意図がよく掴めなかったが、ようは「完全二分木をちょっとダメにしたやつ」を作る。
・証明とか全然ない重罪人なんだけど、根をちょうど真ん中じゃなくて1つずらした所から始めたら無事AC。
・お気持ちとしては、最初がずれてもその先ちゃんと完全二分木を目指して動けば、
　ほぼよしなにやってくれるはずで、木の高さはK〜K*3まで大丈夫なので、さすがに最初が1つずれたくらいでは
　K*3に収まらないほどうまく合わなくなることはないんじゃないかなぁ、って感じ。。
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

K = INT()
N = 2**K

ans = []
visited = [0] * N
def rec(cur):
    if visited[cur]:
        return
    visited[cur] = 1
    ans.append(cur)
    l = cur // 2
    r = (cur+N) // 2
    if l > 0:
        rec(l)
    if r < N:
        rec(r)

# 完全二分木から根を1だけずらしたやつにしてみる
rec(N//2+1)
print(*ans)
