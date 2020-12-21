# -*- coding: utf-8 -*-

S = input()

if S[0] == "A" and S[2:-1].count("C") == 1:
    S = list(S)
    for i in range(len(S)):
        if not S[i].islower() and S[i] != "A" and S[i] != "C":
            print("WA")
            exit()
    print("AC")
else:
    print("WA")