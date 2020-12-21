# -*- coding: utf-8 -*-

"""
参考：https://jag-icpc.org/?plugin=attach&refer=2017%2FPractice%2F模擬地区予選%2F講評&openfile=C.pdf
　　　https://www.hamayanhamayan.com/entry/2017/11/21/030501
・蟻本演習2-6-5
・素数、エラトステネスの区間篩
・最大区間幅W=100万,最大R=10億で、O(√R+WloglogR)=...100万+ちょっと？で
　pythonTLEでpypyAC0.5秒。
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
INF = float('inf')
MOD = 10 ** 9 + 7

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

def segment_sieve(l, r):
    """ 区間[l,r)内の素数の個数(エラトステネスの区間篩) """
    from math import sqrt

    MAX = int(sqrt(r))
    is_prime = [1] * (MAX+1)
    is_prime_seg = [1] * (r-l)

    for i in range(2, MAX+1):
        if is_prime[i]:
            for j in range(i*2, MAX+1, i):
                is_prime[j] = 0
            # 素数が見つかったら、区間内にあるその倍数について素数フラグをfalseにしていく
            for j in range(max(2, (l+i-1)//i)*i, r, i):
                is_prime_seg[j-l] = 0
    return sum(is_prime_seg)

def segment_sieve2(l, r):
    """ 区間[l,r)内の各値の素因数の個数(エラトステネスの区間篩) """
    from math import sqrt

    MAX = int(sqrt(r))
    is_prime = [1] * (MAX+1)
    cnt = [0] * (r-l)
    A = [i+l for i in range(r-l)]

    for i in range(2, MAX+1):
        if is_prime[i]:
            for j in range(i*2, MAX+1, i):
                is_prime[j] = 0
            # 素数が見つかったら、区間内にあるその倍数について素因数分解していく
            for j in range(max(2, (l+i-1)//i)*i, r, i):
                while A[j-l] % i == 0:
                    A[j-l] //= i
                    cnt[j-l] += 1
    # 自身が素数であればその分を足す
    for i, a in enumerate(A):
        if a > 1:
            cnt[i] += 1
    return cnt

l, r = MAP()

# 区間内各値の素因数の個数を取得
res = segment_sieve2(l, r+1)
ans = 0
for a in res:
    if is_prime(a):
        ans += 1
print(ans)
