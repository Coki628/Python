# -*- coding: utf-8 -*-

"""
・400点自力AC
・ゲームの考察
・終了状態から戻って
・負け状態を押し付けられるか
・合ってたからいいけど、もっと公式解説みたいに論理的に考察できたらいいな。
　実際に手番戻してって、奇数奇数だと負けっぽいなーってなっただけだから。
"""

import sys

def input(): return sys.stdin.readline().strip()
def ceil(a, b=1): return int(-(-a // b))
def round(x): return int((x*2+1) // 2)
def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD
def lcm(x, y): return (x * y) // gcd(x, y)
def lcm_list(nums): return reduce(lcm, nums, 1)
def gcd_list(nums): return reduce(gcd, nums, nums[0])
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

A, B = MAP()

if A%2==1 and B%2==1:
    No()
else:
    Yes()
