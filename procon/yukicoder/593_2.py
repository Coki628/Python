"""
参考：https://yukicoder.me/problems/no/593/editorial
・9の倍数、11の倍数の判定方法
・N進数に対して、N-1の倍数、N+1の倍数を判定する場合は上記が使える。
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

def digit_sum(S):
    """ 桁和：O(|S|) """

    res = 0
    for s in S:
        res += int(s)
    return res

S = input()

# 9の倍数の判定方法：3の倍数と同じで普通に桁和が9の倍数かどうか
def check3(S):
    return digit_sum(S) % 3 == 0

# 11の倍数の判定方法：偶奇で桁和を分けて、差が11の倍数かどうか
def check5(S):
    asm = bsm = 0
    for i, s in enumerate(S):
        if i % 2 == 0:
            asm += int(s)
        else:
            bsm += int(s)
    return abs(asm - bsm) % 5 == 0

res3 = check3(S)
res5 = check5(S)

if res3 and res5:
    print('FizzBuzz')
elif res3:
    print('Fizz')
elif res5:
    print('Buzz')
else:
    print(S)
