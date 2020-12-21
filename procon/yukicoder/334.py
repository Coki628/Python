"""
・自力AC
・ゲーム、後退解析、再帰探索
・両者同じ動きをするゲームなので、「1つでもFalseが返ればTrue」が使える。
・イメージはすぐできたのに、実装ちょっと時間かかった。こういう再帰は手早く実装できるようにしたいな。
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
MOD = 10 ** 9 + 7
EPS = 10 ** -10

# 門松列かどうかのチェック
def check(a, b, c):
    mn = min(a, b, c)
    mx = max(a, b, c)
    return a != b and b != c and c != a and (b == mn or b == mx)

def rec(S):
    for i in range(N):
        if S>>i & 1:
            continue
        for j in range(i+1, N):
            if S>>j & 1:
                continue
            for k in range(j+1, N):
                if S>>k & 1:
                    continue
                if check(A[i], A[j], A[k]):
                    if not rec(S | (1<<i) | (1<<j) | (1<<k)):
                        if S == 0:
                            ans.extend([i, j, k])
                        return True
    return False

N = INT()
A = LIST()

ans = []
if rec(0):
    print(*ans)
else:
    print(-1)
