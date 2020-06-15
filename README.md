# Bangdream-difficulty-predictor
邦邦乐曲难度预测器

运行环境：

- Python 3.6.10
- tensorflow 1.15.0 (GPU)
- Keras 2.3.1
- numpy、matplotlib

train.py：用于K折验证，训练500轮后输出平均损失值来确定最佳训练轮次，如果只是想要预测结果的话可以无视。

train2.py：用于实际训练后输出预测值。

predict_xxx：训练后测试集上的输出结果，使用不同的优化器。

> 大致训练流程：
> 从csv文件中的前281首歌曲信息作为测试集，后1000首歌曲信息作为训练集。
>
> 以每首歌曲的时间（Time）、得分比率（Score）、效率（Eff）、每分钟节拍数（BPM）、音符数（N）、每秒钟音符数（NPS）、技能依赖度（SR）作为输入，难度（R）作为输出。
>
> 在输入前需要对输入数据进行标准化，网络结构为2层64个神经元的Dense全连接层+Relu，最后一层为1个神经元的Dense层。

例如下面是对“ヒトリノ夜”四个难度（EX到EASY难度）的预测结果：

[[24.927454]

 [16.998707]

 [13.436136]

 [ 8.783671]]

实际难度值为[25,17,14,9]





代码改编自*Deep learning with Python*

训练数据来自https://bestdori.com/info/songmeta