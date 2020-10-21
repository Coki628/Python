"""
・自力AC！
・実験、回文、文字列
・謎すぎて全く方針が見えてこないのでとりあえず実験してみる。
　するとそれっぽい規則性が見えてくる。
・とりあえずxxx00000xxxみたいな形を取ってる。xxxが増えると0が減る。右と左のxxxは同じ値。
・じゃあxxxを全探索できたら、って思ったらそこは最大9桁にまでなりそうなので無理。
・もう少しよく見ると、xxx自体も回文になってる。なのでxxxの前半分だけ全探索を考える。
　これだとちょうど10000万くらいで、真ん中部分の0〜9合わせても10万ならいけそう、となる。
・文字列⇔数値のキャストが多いから計算量心配だったけど、案外全然大丈夫でpythonAC0.16秒。
　なぜかこの問題メモリ制限が64MBと極端に少なくて、pypyMLEだったのでpythonで。
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
# sys.setrecursionlimit(10**9)
# INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

N = INT()

def check(d):
    d = str(d)
    n = len(d)
    for i in range(n//2):
        if d[i] != d[n-i-1]:
            return False
    return True

# M = 1000000001
# ans = []
# cur = M
# while cur <= N:
#     if check(cur):
#         ans.append(cur)
#     cur += M
# print(len(ans))

ans = 0
for i in range(1, 10000):
    s = str(i)
    rev = s[::-1]
    ln = len(s)*2
    zeros = '0'*(9-ln)
    num = int(s+rev+zeros+s+rev)
    if num <= N:
        ans += 1
    zeros = '0'*(9-ln-1)
    for j in range(10):
        t = str(j)
        num = int(s+t+rev+zeros+s+t+rev)
        if num <= N:
            ans += 1
zeros = '0'*8
for j in range(1, 10):
    t = str(j)
    num = int(t+zeros+t)
    if num <= N:
        ans += 1
print(ans)
