# -*- coding: utf-8 -*-

SA = list(input())
SB = list(input())
SC = list(input())

nxt = "a"
while True:

    if not eval("S" + nxt.upper()):
        print(nxt.upper())
        exit()

    nxt = eval("S" + nxt.upper()).pop(0)