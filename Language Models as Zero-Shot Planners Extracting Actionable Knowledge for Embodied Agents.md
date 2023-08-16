# 【论文笔记】Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents



## Abstract

1. 作者想要实现什么效果？

   本文研究了如何将用自然语言（例如“做早餐”）表达的**高级任务**以基于**一组可操作步骤**（例如“打开冰箱”、“准备食物”、“烹饪”等）进行实现的可能性。

2. 之前都是采用什么方法？

   主要集中在从**明确的逐步操作示例**中学习如何行动。

3. 作者打算采用什么思路？

   预训练的语言模型**足够大**并且**得到适当的提示**，它们可以有效地**将高级任务分解为中级计划**，而无需进一步的训练。

4. 这样的思路有什么问题？

   LLMs生成的计划通常**无法精确映射到可接受的动作**。

5. 作者如何解决这些问题？

   提出了一种程序，它以现有示范/演示为条件，将计划在语义上转换为可接受的动作。

6. 如何评估自己的方法？

   VirtualHome

7. 作者的结论是什么？

   提出的方法大大提高了**可执行性**，超过了LLMs基线。

   所进行的人工评估揭示了**可执行性**和**正确性**之间的权衡。

   从语言模型中**提取可操作知识**是有希望的。

8. 开源网站？

   https://huangwl18.github.io/language-planner

##  Introduction

> <img src="https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/llm_plan_1.png?raw=true" alt="1" style="zoom:50%;" />
>
> 大模型可以产生与人类无差异的行动计划，但通常不能在环境中执行。使用作者的技术，可以显著提高可执行性，<font color=red>尽管以正确性为代价</font>。

---

作者提出了三个猜想：

1. 是否可以将LLMs中包含的这种知识用于目标驱动的决策，以在交互式的、具有体现性的环境中实施。
2. 是否LLMs已经包含了实现目标所需的信息，而无需进行任何额外的训练。
3. 是否可以将关于如何执行高级任务的世界知识扩展到一系列可在环境中执行的可接地面的动作。

作者简单的描述了自己如何设计实验：

- 实验平台： [VirtualHome](http://virtual-home.org/) —— 可以在家庭环境中模拟各种逼真的人类活动，并支持通过使用**动词-宾语**语法定义的具体行动来执行它们的能力。

- 实验评估：依赖人类评估（在 [Mechanical Turk](https://www.mturk.com/) 上进行）来决定行动序列是否有意义地完成了所提出的任务。

- 实验模型：GPT-3、CodeX + 提供单个固定的任务描述示例及其相关的行动序列。

  > 作者说这样算 zero-shot 但是 “提供单个固定的任务描述” 不就是 one-shot 了吗？

- 实验结论：生成在语义上是正确的规划，但生成的行动计划通常无法在环境中执行。生成的行动可能无法精确映射到可接受的行动，或者可能包含各种语言歧义。

作者通过什么方法优化LLMs的规划能力？

1. 列举所有可接受的行动，并将**模型的输出短语**映射到**最具语义相似性**的**可接受行动**。

   How to translate texts to motions?

   这是两个大模型中的一个， named Pre-trained **MASK** LLM 。

2. 使用该模型通过将已**通过上述技术**使其**可接受的先前行动作为条件**，**自回归地生成**计划中的行动。

   How to make the LLM generate better plans?
   
   $$
   A_{t+1} = f_{casual}(s_{t},a_{t},\cdots,a_{1})
   $$
   
   $$
   a_{t+1} = f_{mask}(A_{t+1})
   $$
   
   通过向**模型提示**与**查询任务类似的**已知任务示例，向模型提供**弱监督**。

4. 对多种技术和模型进行人工评估。

---

## Method

> ![2](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/llm_plan_2.png?raw=true)
>
> 预训练因果LLMs（左边）可以将高级任务分解为合理的中级行动计划；
>
> 预训练掩码LLMs（中间）将每个步骤转换为可接受的动作。

### Querying LLMs for Action Plans

上下文信息作为输入提示的一部分提供，并要求LLMs完成其余文本。它通常由自然语言指令和/或包含所需输入/输出对的多个示例组成。

做法：将演示集中的**一个示例高级任务**及**其注释的行动计划**添加到待询问任务的前面。

> 将这个LLMs称为计划LLMs，使用这个LLMs计划生成的方法称为 `Vanilla LLMs` ，其中LLMs被特定的语言模型（如GPT-3）所取代。

---

如何让输出的计划更好？ —— 为每个询问事件生成多个输出，仅考虑每个任务允许评估一个输出样本的情况，这是因为 *重复的试错等价于探索环境以获取特权信息* ，这在我们的设置中不应被视为可行的。

对于 `Vanilla LLMs` ，为了从 $k$ 个样本 $(X_1,X_2,\cdots,X_k)$ 中选择最佳行动计划 $X^{\ast}$，每个样本由 $n_i$ 个tokens组成，选择**平均对数概率最高**的样本。

$$
\arg\max_{X_{i}}  P_{\theta}  (X_{i}) = \frac{1}{n_{i}} \sum_{j=1}^{n_{i}}  \log  p_{\theta}(x_{i,j}|x_{i,\lt j})
$$

### Admissible Action Parsing by Semantic Translation

语义正确但不可行的原因：用**自由形式语言表达的计划**通常无法映射到**明确的可执行步骤**。

1. 输出不遵循任何atom动作的预定义映射。
2. 输出可能使用环境**不可识别的单词**引用atom动作和对象。
3. 输出包含词汇上的歧义。

再次利用语言模型学习的世界知识来翻译语义化动作。

对于每个可接受的环境动作 $a_e$ ，通过**余弦相似度**计算它与**预测的动作短语** $\hat{a}$ 之间的**语义距离**。

$$
C(f(\hat{a}), f(a_e)) := \frac{f(\hat{a})\cdot f(a_e)}{||f(\hat{a})||\cdot ||f(a_e)||}
$$

其中， $f$ 表示词嵌入。使用一个以 `Sentence-BERT` 为目标预训练的BERT风格语言模型，将其称为 `Translation LLMs` 。

###  Autoregressive Trajectory Correction

LLMs可能会为**单个步骤**输出**复合指令**，即使在环境中不能使用一个可接受的动作完成。

可以交替生成计划和翻译动作，以允许自动轨迹校正。

$$
A_{t+1} = f_{casual}(s_{t},a_{t},\cdots,a_{1}) \\
a_{t+1} = f_{mask}(A_{t+1})
$$

在每个步骤中，首先询问 `Planning LM` 来为单个动作生成 $k$ 个样本 $(\hat{a}_1, \hat{a}_2, \cdots, \hat{a}_k)$ 。对于每个样本 $\hat{a}$ ，考虑其**语义合理性**以及在环境中的**可实现性**。具体而言，旨在通过修改排名方案来找到可接受的环境动作 $a_e$ 。

$$
\arg\max\limits_{a_e}\big[\max\limits_{\hat{a}}C(f(\hat{a}), f(a_e)) + \beta· P_{\theta}(\hat{a})\big]
$$

> 先找到LLMs输出的与实际语义最接近的计划动作 $\hat{a}$ ，然后找到这个计划动作 $\hat{a}$ 最接近的动作语义 $a_{e}$ 。

可以使用 `Translation LLMs` 来检测超出分布的动作，即机器人能力范围之外的动作，并及早终止程序，而不是映射到错误的动作。终止的条件是： 

$$
\max\limits_{\hat{a}}C(f(\hat{a}), f(a_e)) + \beta· P_{\theta}(\hat{a})\lt t
$$ 

和百分之 $50$ 当前的采样终止了。

### Dynamic Example Selection for Improved Knowledge Extraction

为了在推理时提供弱监督，从演示集中选择最相似的任务 $T$ 及其示例计划 $E$ ，以用作提示中的示例。

具体而言，重复使用 `Translation LLMs` ，并选择 $(T, E)$ ， $T=\arg\max\limits_{T}C(f(T), f(Q))$ ， $Q$ 是查询任务。

## Evaluation Framework

### VirtualHome

它模拟了在家庭环境中复杂的人类活动。

为了衡量生成的动作计划的正确性，对于这些开放式任务，计算评估本质上是困难的，因此我们进行了类似于VirtualHome论文内的人类评估。

> ![3](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/llm_plan_3.png?raw=true)

---

### Executability

**【可行性定义】可执行性衡量一个动作计划是否可以<font color=red>被正确地解析</font>并<font color=red>满足环境的常识约束</font>。为了满足常识约束，每个动作步骤都必须满足前提条件和后置条件。**

### Correctness

**【语义正确性】<font color=green>自然语言任务模糊范式</font>和<font color=green>多模态特性</font>使得获得正确性的标准测试方法变得不切实际。因此，我们对主要方法进行人类评估。依赖于一种基于匹配的度量方法，该方法衡量<font color=red>生成的程序</font>与<font color=red>人类注释</font>之间的相似程度。具体而言，模仿VirtualHome论文的方法，计算两个程序之间的最长公共子序列（Longest Common Sequence），并将其归一化为两者的最大长度。**

---

## Results

- 比较小的模型（GPT-2）生成更短的计划，但是会被人类误解成正确的、可执行的规划，尽管与实验要求的不符。

- 尽管较小的模型似乎具有更高的可执行性，但是研究发现：大多数这些可执行计划是通过**忽略询问任务**并**重复给定的不同任务示例**来产生的。

  这一点得到了验证，因为较小的模型具有较低的最长公共子序列，尽管具有较高的可执行性。

  表明这种失败模式在较小的模型中普遍存在。

  相比之下，较大的模型不会严重受此失败模式的影响。然而，由于生成的规划语言更具表现力，它们生成的程序实际可执行性显著降低。

- 首先， `Translation LLMs` 在将**复合指令**映射到**简洁的可接受操作**方面表现不佳。

  其次，生成的程序有时会过早终止。这在一定程度上是由于环境表现力的不完善。某些必要的操作或对象未被实现，以完全实现某些任务。

## Analysis and Discussions

1. 消融实验：忽略这三个组件中的任何一个都会导致可执行性和最长公共子序列的性能下降。

   省略动作翻译会导致最显著的可执行性下降，表明从LLM中提取可执行动作计划时动作翻译的重要性。

2. GPT-2 可以生成高度可执行的动作计划，但这些可执行计划大多数是不正确的。

3. 没有观察到 BERT 和 RoBERTa 的不同变体之间在可执行性和最长公共子序列方面存在显着差异。

   假设这是因为在合理大型数据集上训练的任何语言模型**都应该能够执行**本研究中考虑的**单步动作短语翻译**。

4. 之前的研究通常专注于将逐步说明翻译为可执行程序。

   这需要提供任务名称和如何说明以解决多个解决方案的歧义。

   为了调查预训练的LLMs是否可以在不进行额外训练的情况下完成此任务，在提示中包含了这些说明，并使用所提出的方法评估LLMs。

   将结果与 VirtualHome 中从头开始训练人类注释数据的 LSTM 的监督基线进行比较。

   结果显示：即使没有在任何领域数据上进行微调， `Translated Codex` / `GPT-3` 也可以获得接近监督方法的最长公共子序列，并生成高度可执行的程序。

5. 较小的语言模型（如GPT-2）往往生成比较短的程序，同时会频繁重复给定的可执行示例。

   较大的语言模型（如Codex、GPT-3）可以生成更具表现力和高度逼真的程序，但是它们往往会受到可执行性的影响。

## Related Works

- 将语言指令解析为形式逻辑，或者主要依靠词汇分析来解决具身智能体运行的程序中的各种语言歧义。
- 为了在这一领域取得进一步进展，许多学者努力创造了更现实的环境。
- 创建可以执行机器人操作，机器人导航，或两个结合的指令跟踪智能体。
- 最近的研究还使用语言作为分层抽象，以使用模仿学习来规划动作，并在强化学习中引导探索。
- <font color=red>很少有研究在**具体实现**中评估LLMs，这些模型已经通过**在大规模非结构化文本上**进行预训练而包含了行动知识的全部潜力。评估的任务通常是从少数模板中生成的，这些模板与人类在日常生活中执行的高度多样化的活动不相似。</font>

## Conclusion, Limitations & Future Work

1. 正确性的降低 —— 虚拟仿真环境的不充分表达。
2. 有效的高级规划分解 —— 生成的计划必须满足所有常识约束（由可执行性度量所描述）。
3. 是否考虑环境的上下文 —— 在某种程度上，与VirtualHome相同的方式处理LLMs，即**通过想象**，要求**人类注释员**为给定的**人类活动编写行动计划**，在这种情况下，人类同样不观察环境背景。（存在问题）
4. 评估方法 —— 据我们所知，由于任务的开放式和多模态性质，目前还没有一种已知的方法来计算评估计划的语义正确性。以前的工作也采用了类似的指标组合。

## 参考论文

- On the opportunities and risks of foundation models.
- Speaker follower models for vision-and-language navigation.
- Pre-trained transformers as universal computation engines.
- Grounding language in play.
- Language conditioned imitation learning over unstructured data.
- Improving vision-and-language navigation with image-text pairs from the web.
- Exploration through learned language abstraction.
- Virtualhome: Simulating household activities via programs.
- Skill induction and planning with latent language.
- Multimodal few-shot learning with frozen language models.
- Reinforced cross-modal matching and self-supervised imitation learning for vision-language navigation.
