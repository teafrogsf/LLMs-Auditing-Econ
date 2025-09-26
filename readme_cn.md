## Notation

- $T$ : 总任务数量
- $K$: 供应商数目
- $L$: 所有模型、所有任务上：最大的 output tokens
- $b$: 单个任务最大的 reward，因为我们单个任务的metric score 在 0-1 之间，reward = score * reward_param，所以这个值就等于 reward_param，所以这里有一个问题：其实我们直接指定 b 的值就可以了，不用额外再来一个 reward_param 了
- $\mu_i^r$: 每一个模型的期望 reward，其实这个是针对模型而言的，但是在算法描述中容易被理解成是针对供应商而言的，计算公式：
$$
\mu_i^r = \mathbb{E}(\text{score}) * \text{reward\_param}
$$

- $\mu_i^r$: 每一个模型期望的输出长度
- $p_i$: 供应商提供的价格表，price per token

## 流程

### Phase one: Exploration phase

均匀的给每一个供应商 $B$ 次委托，计算每一个供应商的平均 reward，平均 output_tokens，平均 utility，计算方法如下：


计算每一个供应商的 utility，计算平均记作 $u_i$，对于单个供应商，$u_i$ 的计算公式如下:

$$
u = \sum_{j=1}^{B}\frac{1}{B} (\text{reward\_param} * \text{score}_j - \text{price})
$$


## 不诚信

### model 不诚信

供应商的应该怎么做：使用较差的模型来回复，但是报价报最高。
这里有一个很抽象的点

## 9.25 更改
1. 为 `game manager` 添加了更多的 log 输出
2. 更改了 `lie_run_model` 方法的模型选择逻辑：当所有模型的utility都高于second benst utility时，选择provider utility最大的模型，而不是直接选择最差的模型
3. 更改了 `phase 3` 的委托逻辑，现在为：首先直接委托 $ B * int(delta) $ 次；其次委托 $\left \lfloor  B*(delta - int(delta))\right \rfloor$,在此将 $delta - int(delta)$ 命名为frac_part; 最后以B * frac_part - int(self.B * frac_part) 概率进行1次委托
4. 将nl文件夹下的数据更换成2000条的数据（未改文件名，只换了内容）