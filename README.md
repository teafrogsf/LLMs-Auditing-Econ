# PHYbench

## 环境
建议使用linux环境，运行前：`pip install sympy numpy latex2sympy2_extended timeout_decorator`

phybench文件夹是我在ubuntu上使用了venv虚拟环境产生的（如果用conda可以忽略）使用方法为`source phybench/bin/activate`

## 文件说明
`debug_test.py` 用于测试数据集中随机一个数据得到的结果，我调试用的文件，因为真的很好用于是保留
`phybench_evaluation.py` 真正的测试模型的文件，目前输出格式比较混乱，在调整，但是能用

## 其他
作者回消息说他们目前在优化并集成，说不定有脚本嗯嗯，我在等待