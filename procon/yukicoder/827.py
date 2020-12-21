"""
・自力ならず。。全然方針立たなかった。。
・グラフ、木上の数え上げ、確率、期待値の線形性
・期待値の線形性、って言うと仰々しいけど、ようは頂点毎に「ある頂点が神童数になれる確率」
　が求まれば、全通り数*確率でその頂点についての通り数が出せるから、
　あとはそれを全頂点についてやって和を取れば答え、って感じかと。
・考察が立てば、実装はグラフとMOD数え上げの定型をほぼ貼るだけでいい。数え上げの考察むずい。。
・通り数でまず考えてから確率に行くことはちょくちょくあるけど、
　逆に、先に確率で考えて通り数に持って行く方が有効な時もあるんだなー。
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

class ModTools:
    """ 階乗・逆元用のテーブルを構築する """

    def __init__(self, MAX, MOD):

        # nCrならn、nHrならn+rまで作る
        MAX += 1
        self.MAX = MAX
        self.MOD = MOD
        factorial = [1] * MAX
        factorial[0] = factorial[1] = 1
        for i in range(2, MAX):
            factorial[i] = factorial[i-1] * i % MOD
        inverse = [1] * MAX
        inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
        for i in range(MAX-2, -1, -1):
            inverse[i] = inverse[i+1] * (i+1) % MOD
        self.fact = factorial
        self.inv = inverse

    def div(self, x, y):
        """ MOD除算 """

        return x * pow(y, self.MOD-2, self.MOD) % self.MOD

def dfs(nodes, src):
    """ DFS(木、スタック、重みなし) """

    N = len(nodes)
    stack = [(src, -1)]
    dist = [INF] * N
    dist[src] = 0
    while stack:
        u, prev = stack.pop()
        for v in nodes[u]:
            if v != prev:
                dist[v] = dist[u] + 1
                stack.append((v, u))
    return dist

N = INT()
mt = ModTools(N, MOD)
nodes = [[] for i in range(N)]
for _ in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

depth = dfs(nodes, 0)

ans = 0
for i in range(N):
    # 全体の通り数 * ある頂点についての確率 : N! * 1/(depth + 1)
    ans += mt.div(mt.fact[N], depth[i]+1)
    ans %= MOD
print(ans)
