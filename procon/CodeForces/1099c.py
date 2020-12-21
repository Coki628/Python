# -*- coding: utf-8 -*-

S = list(input())
K = int(input())

S2 = []
for i in range(len(S)):
    if S[i] == '?' or S[i] == '*':
        S2[i-1] = ''
        S2.append('')
    else:
        S2.append(S[i])

ln = len(''.join(S2))
if K < ln:
    print('Impossible')
    exit()

for i in range(len(S2)):
    if K == ln:
        print(''.join(S2))
        exit()
    if S[i] == '*':
        S2[i-1] = S[i-1] * (K - ln)
        print(''.join(S2))
        exit()
    if S[i] == '?':
        S2[i-1] = S[i-1]
        ln += 1

if K == ln:
    print(''.join(S2))
else:
    print('Impossible')
