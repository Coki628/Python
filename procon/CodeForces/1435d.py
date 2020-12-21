"""
・自力AC！
・逆操作、シミュレーション、スタック
・スタックっぽく逆順にシミュレーションしていって条件を満たせない場所があればNGとする。
・問題文より、意地悪なケースがだいたいクリアされているので、素直にやってもうまくいった。
・雰囲気でやってしまったので、NG条件漏れてないか心配だったけど無事通ってよかった。。
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

N = INT()
Q = []
for i in range(N*2):
    Q.append(input())

Q.reverse()
stack = []
ans = []
for q in Q:
    if len(q) == 1:
        # 売ったよりも後に、置く操作が来るようなものはNG
        if not stack:
            NO()
            exit()
        a = stack.pop()
        ans.append(a)
    else:
        _, a = q.split()
        a = int(a)
        # 直後に置く予定のもの(スタックの末尾)よりも大きいものが来てしまったらNG
        # (安い順に売れたことにならないため)
        if stack and stack[-1] < a:
            NO()
            exit()
        stack.append((a))

YES()
print(*ans[::-1])
