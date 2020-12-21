"""
・自力AC
・メモ化再帰、辞書順最小の構築
・制約小さいのでDPする。O(K)の貪欲があるらしいが。。
・前からやったら復元だるそうなので再帰で。これなら前から順に取る方優先で潜って、
　初めて当たりが出たやつに確定で戻る時に答えに値追加していけばOK。
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

N, D, K = MAP()

ans = []
visited = set()
def rec(i, d, k):
    if (i, d, k) in visited:
        return False
    if d == D and k == K:
        return True
    if d > D or k > K or i > N:
        return False
    # このお菓子を買う
    if rec(i+1, d+i, k+1):
        ans.append(i)
        return True
    # 買わない
    if rec(i+1, d, k):
        return True
    visited.add((i, d, k))
    return False

if rec(1, 0, 0):
    ans = ans[::-1]
    print(*ans)
else:
    print(-1)
