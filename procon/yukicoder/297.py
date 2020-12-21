"""
参考：https://yukicoder.me/problems/no/297/editorial
・自力ならず。。
・貪欲
・結局、最初に頑張ってた方針が近かった。
・場合分けは合ってたけど、最小値の-がない場合の最適の取り方が正しく作れてなかった。
　確かにマイナス1つも作れないなら、数字固めないで極力バラすが最適だよね、言われればね。。
・他の2パターンが合ってただけに残念だけど、まあこういう適切な貪欲詰めなさい系は
　当たる時は当たるけど無理な時はほんと無理だからしょうがないっちゃしょうがない。
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
sys.setrecursionlimit(10**9)
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

N = INT()
A = []
add = sub = 0
for s in input().split():
    if s == '+':
        add += 1
    elif s == '-':
        sub += 1
    else:
        A.append(int(s))

A.sort(reverse=1)
N = len(A)
blank = N - add - sub - 1
M = N + add + sub + blank
ans1 = ''
j = 0
blankbk, addbk, subbk = blank, add, sub
for i in range(M):
    if i % 2 == 0:
        ans1 += str(A[j])
        j += 1
    else:
        if blank:
            blank -= 1
        elif add:
            ans1 += '+'
            add -= 1
        else:
            ans1 += '-'
            sub -= 1
ans1 = eval(ans1)

blank, add, sub = blankbk, addbk, subbk
if sub == 0:
    # 演算子+を挟んで均等に分ける
    L = add + 1
    B = [''] * L
    for i in range(N):
        B[i%L] += str(A[i])
    ans2 = 0
    for b in B:
        ans2 += int(b[::-1])
else:
    ans2 = str(A[-1]) + '-'
    sub -= 1
    j = 0
    for i in range(2, M):
        if i % 2 == 0:
            ans2 += str(A[j])
            j += 1
        else:
            if blank:
                blank -= 1
            elif sub:
                ans2 += '-'
                sub -= 1
            else:
                ans2 += '+'
                add -= 1
    ans2 = eval(ans2)

print(ans1, ans2)
