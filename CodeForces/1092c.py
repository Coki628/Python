# -*- coding: utf-8 -*-

N = int(input())
sN = [input() for i in range(2*N-2)]

ans_s = ''
ans_ps = ''
def dfs(s, ps, depth):

    global ans_s
    global ans_ps
    if ans_s != '':
        return
    if depth == len(sN):
        ans_s = s
        ans_ps = ps
        return

    s2 = list(s)
    pref = sN[depth] + '*' * (N-len(sN[depth]))
    ok = True
    for i in range(len(pref)):
        if s[i] != pref[i] and s[i] != '*' and pref[i] != '*':
            ok = False
            break
        if s[i] == '*' and pref[i] != '*':
            s2[i] = pref[i]
    if ok:
        dfs(''.join(s2), ps+'P', depth+1)

    s2 = list(s)
    suff = '*' * (N-len(sN[depth])) + sN[depth]
    ok = True
    for i in range(len(suff)):
        if s[i] != suff[i] and s[i] != '*' and suff[i] != '*':
            ok = False
            break
        if s[i] == '*' and suff[i] != '*':
            s2[i] = suff[i]
    if ok:
        dfs(''.join(s2), ps+'S', depth+1)

dfs('*' * N, '', 0)
print(ans_ps)
