"""
np.random.normal()的意思是一个正态分布，normal这里是正态的意思

numpy.random.normal(loc=0.0, scale=1.0, size=None)
loc：float
    此概率分布的均值（对应着整个分布的中心centre）
scale：float
    此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
size：int or tuple of ints
    输出的shape，默认为None，只输出一个值


我们更经常会用到的np.random.randn(size)所谓标准正态分布（μ=0,σ=1），对应于np.random.normal(loc=0, scale=1, size)。
"""
noise = np.random.normal(0, 0.05, x_data.shape)







