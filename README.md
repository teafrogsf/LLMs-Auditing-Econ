## 环境
 `conda env create -p ./env_llm_auditing_econ` 将生成一个叫做`env_llm_auditing_econ`的conda 环境 ，使用`conda activate env_llm_auditing_econ`激活此环境来运行代码。
 
## nl_graph
在此文件夹中，包含了不同NLGraph问题的文件夹，如max_flow,shortest_path问题等。这些文件夹提供了用于评估模型的代码。以max_flow文件夹举例，文件夹包括：

max_flow/

├── cot_prompt.txt

├── max_flow_generator.py

├── max_flow_solver.py

├── main.json

└── max_flow_evaluator.py

其中`max_flow_generator.py`用于生成最大流问题所用图，`max_flow_solver.py`用于形成prompt以及评估LLM的回答，`main.json`为已经生成好的最大流问题数据集，`max_flow_evaluator.py`为其他文件使用一个完整的图生成->评估LLM回答->得到评估分数及其他数据提供接口

## simulator
机制实现的具体代码，通过`experiment.py`运行机制：
`python LLMs-Auditing-Econ/simulator/experiment.py`
机制具体设计见论文，具体函数介绍见文件夹内README.md

## model_evaluator
用于接入不同的测试
提供接口`def evaluate_model(self, model_name: str, test_sample: Dict) -> Tuple[float, int, int]`,参数为要测试的模型名称，以及test sample，返回该模型此次测试得分，input tokens数量以及output tokens数量
