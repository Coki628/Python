"""
・自力AC
・こどふぉAとかで出そうな数学なぞなぞみたいなやつ。
　サンプルが弱すぎて内容理解に時間かかった。。
　(まあこれサンプルが強いとみんなエスパーしてしまうからだろうけど。。)
・大きい方から貪欲に詰める。*2してN以下になるまで。
　お気持ちとしては、大きいやつなら、自分や自分より大きい数と足し合わせても、
　自分*2以下と被ることがないから。
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

N = INT()

ans = []
for i in range(N, -1, -1):
    if i*2 > N:
        ans.append(i)
print(*ans)
