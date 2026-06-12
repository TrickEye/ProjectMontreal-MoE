# Reasoning at the Forking Paths: Identifying and Activating Specialized Experts in Mixture of Experts Models via High-Entropy Token Analysis

总的目标是将一个MoE模型高效地适配到复杂的推理任务：因为预训练的负载均衡和微调的精细化专业化之间的冲突，标准的微调方式不是最优的。 auxiliary load balancing loss 不鼓励专家分化和严重的专家偏好，这导致不同的专家之间存在重叠。因此R-SEFT首先克服这个冲突，然后用数据导向的方法来识别专家的专业。

## Forking Path Hypothesis

这个假设是说多步推理的计算压力 computational effort 在思考过程中不是均匀分布的，而是在模型需要做一个逻辑上的跳跃或决定的关键交界点 forking paths 。这种分叉 token 可以通过模型下一 token 预测概率分布的高不确定性来识别。

LLM 在推理过程中的错误常常出现在高熵 token 处，这些高熵通常对应开启新分支的逻辑连接词（therefore, let, if, because），而低熵 token 对应确定的续写。

在步骤 $t$ 处，有token概率分布 $P_{t}$ ，熵定义为 $H(P_{t})=-\sum_{v\in V}p_{t}(v) \log_{2}p_{t}(v)$ ，熵越高模型越不自信。将高熵 token 定义为最高的 20%。

## Router Unmasking

首先是低消将预训练的负载均衡损失。这些专家有可能通过见到大量的预训练语料形成了隐含的专业能力，有的专家可能见到数学或者逻辑相关的文本。因此第一阶段只在对应的推理数据集上微调路由参数，其他的专家保持冻结。这样让路由精确学习到输入到最适合的专家的映射关系。

## Identifying Reasoning Experts with RFAR

第二阶段目标是精准识别对推理最关键的专家。现有方法用一些粗略的指标，像 average gate scores 来定位专家，并不能区分一个专家到底是处理语义的专家还是作逻辑决定的专家。

因此提出 Reasoning Fork Activation Ratio 来测量一个专家在最不确定的地方激活的比例。

对一个 MoE 层 $E_{i}$ ，RFAR 在一个微调数据集中这样计算：

$RFAR(E_{i})={\sum_{t\in T_{fork}}g_{i}(x_{t}) \over \sum_{{t \in T_{{total}}}}g_{i}(x_{t})}$

$g_{i}(x_{t})$ 是对专家 $E_{i}$ 在 $x_{t}$ 的门控分数，$T_{fork}$ 是所有分岔点的 token （对应20%高熵）。高的 RFAR 对应这个专家在模型高不确定性处密切参与，因此可以在 MoE 架构中定位推理的逻辑。

## Reasoning-Expert Fine-Tuning

选取 top-K RFAR 专家，在最后的微调环节只动这 K 个专家的参数，以提高微调的效率