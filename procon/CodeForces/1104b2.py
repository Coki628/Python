# -*- coding: utf-8 -*-

"""
参考：Test: #7
・終了後、通らなかったテストケース見て条件追加
・添字iが0に戻った時にもi-1との比較やっちまってた。
"""

S = list(input())

i = 1
cnt = 0
while i < len(S) and len(S) >= 2:
    if i != 0 and S[i] == S[i-1]:
        S.pop(i-1)
        S.pop(i-1)
        i -= 2
        cnt += 1
    i += 1

if cnt % 2 == 0:
    print('No')
else:
    print('Yes')
