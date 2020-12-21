"""
参考：https://yukicoder.me/problems/no/701/editorial
　　　https://qiita.com/Scstechr/items/c3b2eb291f7c5b81902a
・公式解より
・文字列のランダム生成をやる。
"""

import sys
from random import choices
from string import ascii_lowercase

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

# n文字の英小文字の文字列をランダム生成する
def generate(n):
    return ''.join(choices(ascii_lowercase, k=n))

used = set()
cur = 'a'
ans = ['a']
for i in range(N-1):
    while 1:
        nxt = cur[-1] + generate(19)
        if nxt[-1] != 'n' and nxt not in used:
            break
    used.add(nxt)
    ans.append(nxt)
    cur = nxt
nxt = cur[-1] + generate(18) + 'n'
ans.append(nxt)

for s in ans[1:]:
    print(s)
