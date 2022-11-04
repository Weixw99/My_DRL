#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/20 0020 21:29
# @Author  : Chao Pan  
# @File    : n次方.py
def power(x, n): #如def power (x,n=2) 设置了n的默认值为2
    s = 1
    while n > 0:
        n = n - 1
        s = s * x
    return s
print(power(4, 4500))

