# -*- coding: utf-8 -*-

S = input()

if S[0] == "A" and S[2:-1].count("C") == 1:
    for i in range(len(S)):
        if i == 1 or i == len(S)-1:
            if not S[i].islower():
                print("WA")
                exit()
        elif 2 <= i <= len(S)-2:
            if not S[i].islower() and S[i] != "C":
                print("WA")
                exit()
    print("AC")
else:
    print("WA")