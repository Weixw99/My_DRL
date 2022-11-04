"""

序列解包（for x,y in zip(keys, values):）详解
"""

# （1）
a, b, c = 1, 2, 3
print(a, b, c)

test_tuple = (False, 3.5, 'test')
d, e, f = test_tuple
print(d, e, f)

x, y, z = map(str, range(3))

print(x, y, z)

"""
输出结果：

1 2 3
False 3.5 test
0 1 2
"""

# （2）序列解包也可以适用于列表和字典呢，字典的话默认是对“key”进行操作，
#     如需对“key”-“value”进行操作则需要使用字典的items()方法进行操作。“value”进行操作的话就使用values()进行操作。

# 列表进行解包
a = [1, 2, 3, 5, 6]
b, c, d, f, g = a
print(b, c, d, f, g)

print('---------------------------')

# 字典进行解包，比较重要
test_dicts = {'a': 'x', 'b': 1, 'c': 3}

q, w, e = test_dicts   # 默认输出键
r, t, y = test_dicts.items()  # 键和值都输出，键与值之间用逗号隔开，
i, o, p = test_dicts.values()  # 输出值

print(q, w, e)
print(r, y, t)
print(i, o, p)

"""
输出结果：
1 2 3 5 6
---------------------------

a b c    
('a', 'x') ('c', 3) ('b', 1)
x 1 3

"""

# （3）还可以用序列解包同时遍历多个序列
list_1 = [1, 2, 3, 4]
list_2 = ['a', 'b', 'c']

for x, y in zip(list_1, list_2):  # 这个要理解
    print(x, y)

"""
输出结果：
1 a
2 b
3 c

list_1中的元素4与list_2没有对应的
"""

# (4)使用内置函数enumerate()返回的的迭代对象进行遍历时的序列解包

# 使用enumerate进行遍历（使用.format()进行格式化）
x = ['a', 'b', 'c']
for i, v in enumerate(x):
    print('遍历出来的值的下标是{0},值是{1}'.format(i, v))

"""
输出结果：
遍历出来的值的下标是0,值是a
遍历出来的值的下标是1,值是b
遍历出来的值的下标是2,值是c
"""

