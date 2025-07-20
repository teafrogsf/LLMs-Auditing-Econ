# PHYbench

## 环境
建议使用linux环境，运行前：`pip install sympy numpy latex2sympy2_extended timeout_decorator`

phybench文件夹是我在ubuntu上使用了venv虚拟环境产生的（如果用conda可以忽略）使用方法为`source phybench/bin/activate`

## 文件说明
`debug_test.py` 用于测试数据集中随机一个数据得到的结果，我调试用的文件，因为真的很好用于是保留

`phybench_evaluation.py` 真正的测试模型的文件，目前输出格式比较混乱，在调整，但是能用

update:所有调试信息已经全部注释

## 在其他文件中使用PHYbench评估模型的EED分数
一共需要用到两个在`phybench_evaluation.py`中的函数，分别是`load_dataset`和`test_random_sample`

由于考虑到后续PHYbench会开源更多有answer的数据，这里没有在初始化的时候就把数据集加载好；如果向load_dataset传入模型和None,则自动加载现有带answer的数据集

`test_random_sample`传入想评估的模型和加载好的数据集，它将会在数据集中随机选择一个数据进行评测，最后返回该模型的EED分数

具体使用可以参考`debug_test.py`

## 其他
作者回消息说他们目前在优化并集成，说不定有脚本嗯嗯，我在等待