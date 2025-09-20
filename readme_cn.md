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

