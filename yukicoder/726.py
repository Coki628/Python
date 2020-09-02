"""
・自力ならず。。。
・ゲームDP、後退解析、メモ化再帰、素数判定
・問題のルールを読み取りきれてなかった。開始位置が素数マスだったら即終了なのかと思ったら違って、
　ゲームは普通に開始されて、最初が素数マスだったことは勝敗に全く関係ないってことらしい。。
・例外処理コメントアウトしたらAC。。。
・ちなみに公式解はもっと頭良く考察して解いてた。。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def is_prime(num):
    """ 素数判定 """
    from math import sqrt

    if num < 2: 
        return False
    if num in [2, 3, 5]: 
        return True
    if num % 2 == 0 or num % 3 == 0 or num % 5 == 0:
        return False
    # 疑似素数(2でも3でも割り切れない数字)で次々に割っていく
    prime = 7
    step = 4
    num_sqrt = sqrt(num)
    while prime <= num_sqrt:
        if num % prime == 0:
            return False
        prime += step
        step = 6 - step
    return True

memo = {}
def rec(h, w):
    if (h, w) in memo:
        return memo[(h, w)]
    # 上に進める
    if not is_prime(h+1) and not is_prime(w):
        if not rec(h+1, w):
            memo[(h, w)] = True
            return True
    # 右に進める
    if not is_prime(h) and not is_prime(w+1):
        if not rec(h, w+1):
            memo[(h, w)] = True
            return True
    memo[(h, w)] = False
    return False

Y, X = MAP()

# ここがいらなかった。。
# if is_prime(Y) or is_prime(X):
#     print('First')
#     exit()

res = rec(Y, X)
if res:
    print('First')
else:
    print('Second')
