# -*- coding: utf-8 -*-

SA = list(input())
SB = list(input())
SC = list(input())

nxt = "a"
while True:
    if nxt == "a":
        if not SA:
            print("A")
            exit()
        nxt = SA[0]
        del SA[0]
    elif nxt == "b":
        if not SB:
            print("B")
            exit()
        nxt = SB[0]
        del SB[0]
    elif nxt == "c":
        if not SC:
            print("C")
            exit()
        nxt = SC[0]
        del SC[0]