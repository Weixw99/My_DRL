"""
time time() 返回当前时间的时间戳（1970纪元后经过的浮点秒数）。

time.time()
参数
    .NA

返回值:
返回当前时间的时间戳（1970纪元后经过的浮点秒数）
时间戳：是指格林威治时间1970年01月01日00时00分00秒(北京时间1970年01月01日08时00分00秒)起至现在的总秒数

"""

import time

print("time.time(): %f " %  time.time())
print(time.localtime( time.time() ))
print(time.asctime( time.localtime(time.time()) ))

"""
输出结果：（因为是时间，所以是时变的，以下面举例：）
（1）
time.time(): 1584245731.038167   # 1970年01月01日00时00分00秒(北京时间1970年01月01日08时00分00秒)起至现在的总秒数

（2）
# tm_wday=6是指一周的第几天（索引从0开始，即星期一是tm_wday=0，那现在这个tm_wday=6是星期天）
#  tm_yday=75是指当年从1月1号开始起，第75天
# tm_isdst：
tm_isdst = 1 的时候表示时间是夏令时，

　　　　值为0的时候表示非夏令时

　　　　值为-1的时候表示时间不确定是否是夏令时 

time.struct_time(tm_year=2020, tm_mon=3, tm_mday=15, tm_hour=12, tm_min=15, tm_sec=31, tm_wday=6, tm_yday=75, tm_isdst=0)

（3）
Sun Mar 15 12:15:31 2020


"""

