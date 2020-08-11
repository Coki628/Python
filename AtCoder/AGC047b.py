"""
参考：https://atcoder.jp/contests/agc047/editorial/43
・終了後、ちょっと方針見たあとで自力、TLE。
・まず、題意を満たす(prefix2文字から選んで消す、で得られる)文字列は、
　「prefixから好きに1文字選んだやつ+suffix」になる。
　(ここが考察で出ないことにはスタートラインに立てなそうだ。。)
・扱いやすくするため、逆順にして「prefix+その後ろから好きに1文字選んだやつ」にする。
・あとは後ろに何が残ってるかをCounterとかで管理すると、各文字について、
　「そこより手前+a~zから1文字」が元の文字列のどれかと一致しているかを確認する、で
　1回の操作にかける計算量がO(1)なら、全体の計算量が文字数*26=2600万になって間に合う。
・これは部分文字列の取得が文字列スライスでO(1)じゃないのでTLE。
"""

import sys
from collections import Counter
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
INF = 10 ** 19
MOD = 10 ** 19 + 7
EPS = 10 ** -10

N = INT()
se = set()
for i in range(N):
    s = input()
    s = s[::-1]
    se.add(s)

ans = 0
for s in se:
    C = Counter(s)
    for i in range(len(s)):
        for c in ascii_lowercase:
            if C[c]:
                if s[:i]+c in se:
                    ans += 1
        C[s[i]] -= 1
        if C[s[i]] == 0:
            del C[s[i]]
ans -= N
print(ans)
