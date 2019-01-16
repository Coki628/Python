# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/arc081/editorial.pdf
・DP作って数え上げ、とか考えてしまった。
・こんなシンプルにパターン分けするだけでいけるとは…。
"""

MOD = 1000000007

N = int(input())
S1 = input()
S2 = input()

ptns = []
i = 0
# パターンXとYに分けて区切る
while i < N:
    if S1[i] == S2[i]:
        ptns.append('X')
        i += 1
    else:
        ptns.append('Y')
        i += 2

if ptns[0] == 'X':
    ans = 3
elif ptns[0] == 'Y':
    ans = 6
for i in range(1, len(ptns)):
    # XとYの並びを4つパターン毎に処理
    if ptns[i-1] == 'X' and ptns[i] == 'X':
        ans *= 2
    elif ptns[i-1] == 'X' and ptns[i] == 'Y':
        ans *= 2
    elif ptns[i-1] == 'Y' and ptns[i] == 'X':
        ans *= 1
    elif ptns[i-1] == 'Y' and ptns[i] == 'Y':
        ans *= 3
    ans %= MOD
print(ans)
