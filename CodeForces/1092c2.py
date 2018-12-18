# -*- coding: utf-8 -*-

"""
提出間に合わなかった版
・あと一歩…これでAC通せたのに。
"""

N = int(input())
sN = [input() for i in range(2*N-2)]

ans_s = ''
ans_ps = ''
done = set()
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
    # 同じ文字列を2回Pにしないように対策
    if ok and sN[depth] not in done:
        done.add(sN[depth])
        dfs(''.join(s2), ps+'P', depth+1)
        if sN[depth] in done:
            done.remove(sN[depth])

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
        done.add(sN[depth])
        dfs(''.join(s2), ps+'S', depth+1)
        if sN[depth] in done:
            done.remove(sN[depth])

dfs('*' * N, '', 0)
print(ans_ps)
