"""
pickle.dump(obj, file, [,protocol])

Python中的Pickle模块实现了基本的数据序列与反序列化。通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储；
通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。

解释：

序列化对象，将对象obj保存到文件file中去。参数protocol是序列化模式，
默认是0（ASCII协议，表示以文本的形式进行序列化），
protocol的值还可以是1和2（1和2表示以二进制的形式进行序列化。其中，1是老式的二进制协议；2是新二进制协议）。
file表示保存到的类文件对象，file必须有write()接口，file可以是一个以'w'打开的文件或者是一个StringIO对象，
也可以是任何可以实现write()接口的对象。如果protocol>=1，文件对象需要是二进制模式打开的。
"""

#使用pickle模块将数据对象保存到文件

import pickle

data1 = {'a': [1, 2.0, 3, 4+6j],
         'b': ('string', u'Unicode string'),
         'c': None}

selfref_list = [1, 2, 3]
selfref_list.append(selfref_list)
print(selfref_list)

output = open('data.pkl', 'wb')

print("output为：",selfref_list)
print('\n')

# Pickle dictionary using protocol 0. 使用协议0 Pickle字典
pickle.dump(data1, output)

print("pickle.dump：",pickle.dump)  # 输出built-in function dump，这个意思：内置函数转储
print('\n')
# Pickle the list using the highest protocol available.使用可用的最高协议Pickle列表
pickle.dump(selfref_list, output, -1)

print("下一个pickle.dump为：",pickle.dump)  # 输出built-in function dump，这个意思：内置函数转储
output.close()

