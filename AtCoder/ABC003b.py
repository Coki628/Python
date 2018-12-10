# -*- coding: utf-8 -*-

S = input()
T = input()

def check():
    for i in range(len(S)):
        if S[i] == '@' and T[i] != '@':
            if T[i] not in ['a', 't', 'c', 'o', 'd', 'e', 'r']:
                return False
        elif T[i] == '@' and S[i] != '@':
            if S[i] not in ['a', 't', 'c', 'o', 'd', 'e', 'r']:
                return False
        elif S[i] != T[i]:
            return False
    return True

if check():
    print('You can win')
else:
    print('You will lose')
