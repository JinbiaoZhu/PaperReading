# 【论文笔记】Skill-based Meta Reinforcement Learning

## Abstract

1. 研究背景。

   > While deep reinforcement learning methods have shown impressive results in robot learning, their **sample** **inefficiency** makes the learning of <font color=red>complex</font>, <font color=red>long-horizon</font> behaviors with real robot systems infeasible. 

   虽然深度强化学习方法在机器人学习中表现出色，但它们的样本效率使得在真实机器人系统中学习复杂、长期的行为变得不可行。

   > To mitigate this issue, meta reinforcement learning methods aim to enable fast learning on novel tasks by learning how to learn.

   为了缓解这个问题，元强化学习方法旨在通过学习如何学习，使得在新任务上能够快速学习。

2. 研究现状。

   限制在短期行为、密集奖励函数上。

3. 针对这个问题，作者的研究思路？

   > To enable learning long-horizon behaviors, recent works have explored leveraging prior experience in the form of offline datasets without reward or task annotations.

   为了使学习长期行为变得可能，最近的研究探索了利用以离线数据集形式存在的先验知识，这些数据集没有奖励信息或任务注释。

   > In this work, we devise a method that enables meta-learning on long-horizon, sparse-reward tasks, allowing us to solve unseen target tasks with orders of magnitude fewer environment interactions.

   制作这样的离线数据集需要大量的实践交互。在这项工作中，我们设计了一种方法，可以在长期、稀疏奖励的任务上进行元学习，使我们能够以<font color=red>少得多的环境交互</font>来解决未见过的目标任务。

4. 针对作者的研究思路，作者的创新点和技术路线？

   >Our core idea is to leverage prior experience extracted from offline datasets during meta-learning.

   我们的核心思想是在元学习过程中利用从离线数据集中提取的先前经验。

   - Extract reusable skills and a skill prior from offline datasets.

     从离线数据集中**提取可重用技能**和**技能先验**。

   - Meta-train a high-level policy that learns to efficiently compose learned skills into long-horizon behaviors.

     元训练一个高层策略，学习将从离线数据集抽取到的技能有效地组合成长期行为。

   - Rapidly adapt the meta-trained policy to solve an unseen target task.

     将元训练得到的策略快速泛化适应到新的、未见的目标任务上。

5. 作者的实验设计？

   > continuous control tasks in navigation and manipulation
   >
   > maze navigation and kitchen manipulation

   导航任务（自己设计的）和机器人操作任务（`Meta-World`  框架）

6. 作者的实验指标？

   在摘要中未曾提及。

7. 作者的结论？

   >The proposed method can efficiently solve long-horizon novel target tasks by combining the strengths of meta-learning and the usage of offline datasets, while prior approaches in RL, meta-RL, and multi-task RL require substantially more environment interactions to solve the tasks.

   该方法通过结合元学习和使用离线数据集的优势，可以有效地解决长期新目标任务，而强化学习、元强化学习和多任务强化学习的先前方法需要更多的环境交互才能解决这些任务。

8. 是否开源？

   是，https://namsan96.github.io/SiMPL/

## 1 INTRODUCTION

> In contrast, humans are capable of effectively learning a variety of complex skills in only a few trials. This can be greatly attributed to our ability to learn how to learn new tasks quickly by efficiently utilizing previously acquired skills.

相比之下，人类能够在仅几次试验中有效地学习各种复杂技能。这可以很大程度上归因于我们学习如何快速学习新任务的能力，通过有效地利用先前获得的技能。

> <font color=blue size=2>机器人技能操作这个领域，引用元强化学习的写法都是从人类角度引出来的。</font>
>
> <font color=blue size=2>也可以换一种写法，就是“学者借鉴cv、nlp相关的解决小样本问题的方法，进而引出元强化学习。”</font>

---

作者介绍了近几年开发出来的元强化学习的几个特点：

1. by learning to learn from a distribution of tasks (Finn et al., 2017; Rakelly et al., 2019).

   从一组分布中，学习掌握学习能力（引用的是经典的MAML和PEARL方法）

2. restricted to short-horizon, dense-reward tasks.

   限制在短期的、密集奖励函数的任务上。（`Meta-World` 的奖励函数精确到了4~5位）

---

作者给出了近几年这个方向的发展：

> Recent works aim to leverage experience from prior tasks in the form of offline datasets without additional reward and task annotations.

使用过去任务的经验（以离线数据集的形式、没有奖励信息和任务标注）

作者参考了如下文献：(<font color=blue size=2>看作者和标题感觉很offline的内容</font>)

1. Learning latent plans from play. In Conference on Robot Learning, 2020.
2. Accelerating reinforcement learning with learned skill priors. In Conference on Robot Learning, 2020.
3. Actionable models: Unsupervised offline reinforcement learning of robotic skills. arXiv: 2104.07749.

作者认为这样的离线方法的不足之处：

> While these methods can solve complex tasks with substantially improved sample efficiency over methods learning from scratch, millions of interactions with environments are still required to acquire long-horizon skills.

采用离线的方式可以显著提升样本有效性，但是获得长期的技能仍然还需要很多与环境的交互。

---

- 作者的研究思路：结合元强化学习的学习能力获得和离线强化学习的学习无标注任务离线数据集的能力。

- 作者的研究目标：能够在复杂的、长期的任务上进行元学习，并可以以比以前的工作**少一个数量级的环境交互**来解决看不见的目标任务。(研究的flag不要立的太大hhhhhh)

> ![sbmrl-pic-1](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_1.png?raw=true)
>
> 利用大量的离线数据集，其中包含在许多任务中收集的先前经验，没有奖励或任务注释；
>
> 利用元训练任务，以学习如何快速解决看不见的长期任务。

---

作者对本文的“技能”做诠释：可以被重利用；是短期的行为；能被组合成长期行为。

> reusable skills – short-term behaviors that can be composed to solve unseen long-horizon tasks.

因为作者的立意是短期行为组成长期的行为，那么对于组合优化任务用强化学习是最好的。因此作者采用了<font color=red size=5>分层元强化学习</font>框架，顶层学习如何重利用、排列抽取到的技能。但是从零开始训练这个顶层策略是困难的，作者还是采用了离线的方式，学习技能组合的离线形式的先验知识。

之前作者在Abstract中提到的，可重用技能就是用于组合的技能，技能先验就是组合这些技能的先验知识。

---

作者Flag：这是第一个将元强化学习算法与不包含奖励或任务注释的任务无关的离线数据集相结合的工作。

<font color=blue size=2>作者跟传统强化、多任务强化、元强化学习比较；但是作者没有跟离线强化比较？可以增加一个这样的baseline~</font>

## 2 RELATED WORK

### Meta-Reinforcement Learning

作者列举了一大堆元强化学习的作品，然后提出他们的工作是短期的、密集奖励的；引出自己的工作是长期的、稀疏奖励的任务。

### Offline datasets

作者介绍了一下离线强化学习。<font color=blue size=2>（涉及比较少，这里抛出离线强化的目标——从完全离线的预收集的数据中学习，脱离环境交互）</font>

> In particular, the field of offline reinforcement learning (`Levine et al., 2020; Siegel et al., 2020; Kumar et al., 2020; Yu et al., 2021`) aims to devise methods that can perform RL fully offline from pre-collected data, without the need for environment interactions.

作者接下来分析了离线强化学习的实现前提——标注。

> However, these methods require target task reward annotations on the offline data for every new tasks that should be learned. These reward annotations can be challenging to obtain, especially if the offline data is collected from a diverse set of prior tasks.

这些方法要求在每个应该学习的新任务的离线数据上进行目标任务奖励注释。这些奖励注释可能难以获得，特别是如果离线数据来自各种不同的先前任务。

> 离线强化学习的数据集需要标注是因为它的训练数据是预先收集的，而不是实时生成的，因此需要标注来告诉模型每个状态下采取的行动和获得的奖励是什么。标注的内容包括状态、行动和奖励等信息。
>
> 目前还没有完全无标注的离线强化学习方法，但是有些方法可以减少标注量，比如使用自监督学习、利用环境模型等技术。这些方法可以通过对未来状态的预测来减少标注量，但是需要更多的计算资源和更复杂的算法。
>
> 离线强化学习的数据集基本形式是由一系列状态、行动和奖励组成的序列，这些数据是预先从环境中收集的，而不是实时生成的。数据集的构成方式包括随机数据集、专家数据集、混合数据集、噪声数据集和重放数据集等。

这样的话，作者的工作就很先进了：进行的离线强化学习不需要奖励标注。

### Offline Meta-RL

定义：从静态的，预先收集的数据集，包括奖励注释，进行元学习。

> Another recent line of research aims to *meta-learn* from static, pre-collected datasets including reward annotations.

1. Offline meta-reinforcement learning with advantage weighting. In International Conference on Machine Learning, 2021.
2. Offline meta-reinforcement learning with online self-supervision. arXiv: 2107.03974, 2021.
3. Offline meta learning of exploration. In Neural Information Processing Systems, 2021.

> After meta-training with the offline datasets, these works aim to quickly adapt to a new task with only a small amount of data from that new task.

这样的离线元强化学习的泛化目标是根据待适应的目标域特定任务的有限数据做泛化。

<font color=blue size=2>可以这样理解，元强化学习本身是从元学习引进过来的，那么元学习本身就是实现同一分布不同数据集上的泛化。离线强化学习把在线的经验转变/固化成了一份份数据集，似乎天然适合元学习的泛化适应。</font>

因为形式上非常想元学习跑数据集，那么它的不足之处在于数据集上。

> However, in addition to reward annotations, these approaches often require that the offline training data is split into separate datasets for each training tasks, further limiting the scalability.

元训练和元测试需要划分数据集，元训练甚至还要进一步拆分数据集，这就导致数据集被分的很”散“。

### Skill-based Learning

作者认为：在没有奖励或者任务标注的情况下，使用没有标签的离线数据集的方法，就是技能抽取。<font color=blue size=2>可以这样理解，对于没有样本标签的数据，采用一种类似聚类的方式提取一些信息。在这里这些提取出来的信息就是“技能”。</font>

1. Composing complex skills by learning transition policies. In International Conference on Learning Representations, 2018.
2. Learning an embedding space for transferable robot skills. In International Conference on Learning Representations, 2018.
3. Dynamics-aware unsupervised discovery of skills. In International Conference on Learning Representations, 2020.
4. Program-Guided Framework for Interpreting and Acquiring Complex Skills with Learning Robots. PhD thesis, University of Southern California.

> Yet, although they are more efficient than training from scratch, they still require a large number of environment interactions to learn a new task. Our method instead combines skills extracted from offline data with meta-learning, leading to significantly improved sample efficiency.

通过从离线数据抽取的方式可以让数据更加有效，但是并不意味着它效率很高。因为组合一个长期的任务，还需要很多次的迭代。

## 3 PROBLEM FORMULATION AND PRELIMINARIES

### Problem Formulation

1. 离线数据集。

   形式：<font color=red>状态动作对</font>的轨迹 $\mathbf{D}:\{s_{0},a_{0},\cdots,s_{n},a_{n}\}$ 。

   获得：自动探索、人类遥操作，先前训练好的智能体。

   特点：任务无关（Task-Agnostic）。

2. 元强化学习的任务集。

   目的：用于元训练。

   形式：集合内的每个任务实际都是一个MDP。

   $$\mathbf{T}=\{T_{1},T_{2},\cdots,T_{n}\}$$

4. 问题目标：

   > Our goal is to leverage both, the offline dataset $\mathbf{D}$ and the meta-training tasks $\mathbf{T}$, to accelerate the training of a policy $π(a|s)$ on a target task $\mathbf{T}^{\ast}$ which is also represented as an MDP.

   我们的目标是利用离线数据集 $\mathbf{D}$ 和元训练任务集 $\mathbf{T}$，加速在目标任务 $\mathbf{T}^{\ast}$ 上训练策略 $π(a|s)$ 的过程，该目标任务也被表示为一个 MDP。

   > We do not assume that $\mathbf{T}^{\ast}$ is a part of the set of training tasks $\mathbf{T}$, nor that $\mathbf{D}$ contains demonstrations for solving $\mathbf{T}^{\ast}$ .

   要解决的任务 $\mathbf{T}^{\ast}$ 不属于任务集合 $\mathbf{T}$，此外， 离线数据集 $\mathbf{D}$ 中没有包含要解决的任务 $\mathbf{T}^{\ast}$ 的演示。

### 基于技能的方法如何解决这个问题？

对标方法：SPiRL——Skill Prior RL。

SPiRL 使用任务无关的数据集来训练两个模型：

- a skill policy $π(a|s, z)$ that decodes a latent skill representation $z$ into a sequence of executable actions.

  ”解码“（我的理解是”编码“）将技能编码成一个潜在变量 $z$ 并于当前状态一起送入策略中，获得动作。

- a prior over latent skill variables $p(z|s)$ which can be leveraged to guide exploration in skill space.

  训练一个分布 $p(z|s)$ 实现”在什么状态下采用什么技能最好“。

- SPiRL uses these skills for learning new tasks efficiently by training a high-level skill policy $π(z|s)$ that acts over the space of learned skills instead of primitive actions.

  在新任务上，不使用最本质地动作，而是使用”技能组合“来实现任务的完成。

SPiRL 的目标是：

$$
\max\limits_{\pi}\sum\limits_{t}E_{(s_{t},a_{t})\sim\rho_{\pi}}[r(s_{t},z_{t})-\alpha D_{KL}(\pi(z|s_{t})|p(z|s_{t}))]
$$

### 异策略元强化学习方法如何解决这个问题？

对标方法：PEARL。

PEARL leverages the meta-training tasks for learning a task encoder $q(e|c)$ . This encoder takes in a small set of state-action-reward transitions $c$ and produces a task embedding $e$. This embedding is used to condition the actor $π(a|s, z)$ and critic $Q(s, a, e)$ .

PEARL学习一个编码器  $q(e|c)$ 。采用”状态-动作-奖励“三个联动信息作为上下文 $c$ 来产生任务嵌入  $e$ 。任务嵌入 $e$ 被用来调节/控制策略的输入： $π(a|s, z)$ 以及状态动作对的评价 $Q(s, a, e)$ 。

PEARL 的目标是：

$$
\max\limits_{\pi}E_{T\sim p_{t},e\sim q(\cdot|c^{T})}\big[\sum\limits_{t}E_{(s_{t},a_{t})\sim\rho_{\pi|e}}[r_{T}(s_{t},a_{t})+\alpha H(\pi(a|s_{t},e))]\big]
$$

## 4 APPROACH

> ![sbmrl-pic-2](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_2.png?raw=true)
>
> - **Skill Extraction**: learns reusable skills from snippets of task-agnostic offline data through a skill **extractor (yellow)** and **low-level skill policy (blue)**. Also trains a prior distribution over **skill embeddings (green)**. 
> - **Skill-based Meta-training**: Meta-trains **a high-level skill policy (red)** and **task encoder (purple)** while using the pre-trained low-level policy. The pre-trained skill prior is used to <font color=red>regularize the high-level policy</font> during meta-training and guide exploration.
> - **Target Task Learning**: Leverages the meta-trained hierarchical policy for quick learning of an unseen target task. After conditioning the policy by encoding a few transitions $c^{\ast}$ from the target task $T^{\ast}$ , we continue fine-tuning the high-level skill policy on the target task while regularizing it with the pre-trained skill prior.

---

### 4.1 SKILL EXTRACTION

1. 训练技能编码器 $q(z|s_{0:K}, a_{0:K−1})$，将从数据集 $\mathbf{D}$ 中随机裁剪的 $K$ 步轨迹嵌入到低维技能嵌入 $z$ 中。
2. 低层技能策略 $π(a_{t}|s_{t}, z)$ ，通过行为克隆进行训练，以在给定技能嵌入的情况下重现动作序列 $a_{0:K−1}$ 。
3. 使用单位高斯分布对技能编码器的输出进行正则化，并通过系数 $β$ 对此正则化进行加权。
4. 学习一个技能先验 $p(z|s)$ ，该先验捕捉在训练数据分布下可能在给定状态下执行的技能分布。

技能抽取过程的目标：

![sbmrl-pic-3](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_3.png?raw=true)

> 实际上，对这个 $\pi(a|s,z)$ 做一些推导可以知道为什么作者要训练这些”组件“。
> 
> $$
> \begin{align}
> \pi(a|s,z) =& \frac{p(a,s,z)}{p(s,z)} \text{（条件概率公式拆开）} \\
> =& \frac{p(z|s,a)p(s,a)}{p(z|s)p(s)} \text{（分子分母依次使用全概率公式）} \\
> =& p(a|s)\frac{p(z|s,a)}{p(z|s)} \text{（把分子分母的第二项合并起来）}
> \end{align}
> $$
> 
> 
> 也就是说，得到 $\pi(a|s,z)$ 策略，需要（1） $p(a|s)$ 离线数据集生成的策略，这个当作已知的，可以通过行为克隆获得；（2） $p(z|s,a)$ 根据离线数据集抽取的”技能“，也就是技能抽取器；（3） $p(z|s)$ ，在当前状态下可能采用何种”技能“的概率，也就是先验分布。
> 
> $$
> \begin{align}
> \max\pi(a|s,z)	&=	\max{p(a|s)\frac{p(z|s,a)}{p(z|s)}} \\
> 				&=	p(a|s)\frac{\max p(z|s,a)}{\min p(z|s)}
> \end{align}
> $$
> 
> 可见，分子的最大化对应于目标的第一项；分母的最小化对应于目标的第二项。

---

### 4.2 SKILL-BASED META-TRAINING

思路：模仿 PEARL 的异策略方式，训练元策略。

1. 训练任务编码器。收集一些状态转移的样本，生成任务嵌入向量 $e$ 。

2. 通过训练任务嵌入 $e$ 的条件策略，利用我们的技能，而不是原始行动： $π(z|s,e)$ 。从而用一组有用的预训练”技能“装备，并减少元训练任务，让元训练学习如何结合这些行为而不是从头学习。

3. 元训练阶段的目标：

   ![sbmrl-pic-4](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_4.png?raw=true)

   我们通过在策略和先验之间选择一个目标散度 $δ$，通过双梯度下降自动调优 $α$ 。

4. 我们使用了多种不同大小的 $c$ 。通过调整先验正则化的强度到条件集的大小来增加训练的稳定性。

   > When the set $c$ is small, it has only little information about the task at hand and should thus be regularized stronger towards the task-agnostic skill prior.

---

### 4.3 TARGET TASK LEARNING

作者给出了元测试阶段泛化适应的任务：策略应该首先探索不同的技能选项以了解手头的任务，然后迅速缩小其输出分布到那些解决任务的技能。

> Intuitively, the policy should fifirst explore different skill options to learn about the task at hand and then rapidly narrow its output distribution to those skills that solve the task.

1. We explore the environment by conditioning our pre-trained policy with task embeddings sampled from the task prior $p(e)$ .

   作者首先解决了探索任务：预训练策略 + 从任务先验 $p(e)$ 采样得到的任务嵌入，来探索

2. Then, we encode this set of transitions into a target task embedding $e^{\ast}\sim q(e|c^{\ast})$.

   用元训练好的编码器获得目标域任务嵌入 $e^{\ast}$

3. By conditioning our meta-trained high-level policy on this encoding, we can rapidly narrow its skill distribution to skills that solve the given target task: $π(z|s, e^{\ast})$.

   再把选取好的任务嵌入 $e^{\ast}$ 联合状态加入到策略中，获得当前状态和任务嵌入的技能 $z$

---

作者在此处提到了他们的研究成果：only very few interactions，就能实现目标域上的快速泛化适应。

![sbmrl-pic-5](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_5.png?raw=true)

## 5 EXPERIMENTS

- 我们提出的方法能否学会高效地解决长时间跨度、奖励稀疏的任务？——“验证”
- 利用离线数据集是实现这一目标是至关重要的吗？——“消融”
- 我们如何最好地利用训练任务，以实现目标任务的高效学习？——“调参”
- 训练任务分布如何影响目标任务的学习？——“调参”

### 5.1 EXPERIMENTAL SETUP

> ![sbmrl-pic-6](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_6.png?raw=true)
>
> 1. 智能体需要导航数百步才能到达未见过的目标，并且只有在任务成功时才会获得二进制奖励。
> 2. 7自由度智能体需要执行四个子任务的一个看不见的序列，跨越数百个时间步长，并且只有在完成每个子任务时才能获得稀疏的奖励。

#### 5.1.1 MAZE NAVIGATION

智能体的观测空间包括其二维位置和速度，并通过平面连续速度命令进行操作。

数据集：通过在迷宫中随机采样起点和目标点，并使用规划器生成从起点到目标点的轨迹。没有标注奖励和任务信息。

> To generate a set of meta-training and target tasks, we fix the agent’s initial position in the center of the maze and sample 40 random goal locations for meta-training and another set of 10 goals for target tasks.
>
> <font color=blue size=2>作者的元训练中固定了智能体的起点，元训练和测试的任务采样都比较多一些。还有一个是，这个迷宫的共有信息太多了。很多目标点具有相似的走行路径，比如说元测试是A点到C点，但是A、C之间有个B点，在元训练中得到了有很好表现的A-B策略，那么泛化到A-C任务时，智能体可能先走A-B路径，再泛化B-C路径的策略。这就说明这样的任务场景下，提取信息很容易。也说明了采样的任务太“密集”了。</font>

#### 5.1.2 KITCHEN MANIPULATION

我们利用600个人类远程操作操作序列的数据集进行离线预训练。

### 5.2 BASELINES

| 名称 | SAC        | SPiRL                                                        | PEARL                        | PEARL-finetune | Multi-task RL                                                |
| ---- | ---------- | ------------------------------------------------------------ | ---------------------------- | -------------- | ------------------------------------------------------------ |
| 备注 | 熟悉的基线 | 它先从离线数据集获得一种技能和一种技能先验分布，但不使用元训练任务。 | 最先进的异策略元强化学习算法 | 用SAC做微调    | 通过将每个任务中专用的单个策略提炼为共享策略。它同时利用了与我们的方法相似的元训练任务和离线数据集。 |

### 5.3 RESULTS

While PEARL is first trained on the meta-training tasks, it still achieves poor performance on the target tasks and fifine-tuning it (PEARL-ft) does not yield signifificant improvement.

> We believe this is because both environments provide only sparse rewards yet require the model to exhibit long-horizon and complex behaviors, which is known to be diffificult for meta-RL methods.
>
> <font color=blue size=2>作者的意思是因为环境是稀疏奖励和长视野的，所以PEARL效果不好。</font>
>
> <font color=blue size=2>我感觉这个因果关系不是很明显，没有解释到PEARL的底层里面。</font>

> ![sbmrl-pic-7](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_7.png?raw=true)

> ![sbmrl-pic-8](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_8.png?raw=true)

### 5.4 META-TRAINING TASK DISTRIBUTION ANALYSIS

我们研究了以下两个因素的影响

- 元训练任务分布中的任务数量
- 元训练任务分布与目标任务分布之间的对齐效果

#### Number of meta-training tasks.

采样任务数从20变成了10。

作者的结论是：仍然数据有效。

> SiMPL is still more sample efficient compared to the best-performing baseline (i.e. SPiRL).
>
> <font color=blue size=2>还是保留我的观点，在迷宫环境，共享信息太容易获得，数据肯定有效。</font>

#### Meta-train / test task alignment （更有说服力！）

作者创建了有偏见的元训练/测试任务分布。

从迷宫的顶部 25% 的部分中抽取目标位置来创建元训练集 (TTRAIN-TOP)。为了排除任务分布密度的影响，从中抽取了 10 个 (即 40 × 25%) 元训练任务。然后，我们分别从迷宫的顶部 25% 部分 (TTARGET-TOP) 和底部 25% 部分 (TTARGET-BOTTOM) 中各抽取了 10 个目标任务，创建了两个与该元训练分布具有良好和差的对齐效果的目标任务分布。

> 1. The results demonstrate that SiMPL can achieve improved performance when trained on a better aligned meta training task distribution.
>
> 2. This is expected given that SPiRL does not learn from the misaligned meta-training tasks.
>
>    说明这个模型在分布差异很大的时候，目标分布不一致的时候，表现较差。
>
>    This is expected given that SPiRL does not learn from the misaligned meta-training tasks.
>
>    作者给出的理由是：没有学到有偏差的训练任务。

> ![sbmrl-pic-9](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_9.png?raw=true)

## 6 CONCLUSION

> In the future, we aim to demonstrate the scalability of our method to **high-DoF continuous control** problems on real robotic systems to show the benefifits of our improved sample effificiency.

## APPENDIX

### A META-REINFORCEMENT LEARNING METHOD ABLATION

作者对比：时间上延申出来的技能 和 利用先验知识的方法，哪个对稀疏奖励和长视野的环境更重要？

- **BC+PEARL**：首先，通过离线数据集的监督学习，学习了一个行为克隆（BC）策略。然后，类似于我们的方法 SiMPL，在元训练阶段，使用 BC 策略受限的 SAC 目标元训练了任务编码器和元学习策略。
- **BC+MAML**：把 PEARL 算法取代成 MAML，采用的是MAML+TRPO
- 标准的 PEARL。

---

作者的实验设置：

1. short-range goals with small variance TTRAIN-EASY.
2. short-range goals with larger variance TTRAIN-MEDIUM.
3. long-range goals with large variance TTRAIN-HARD, which we used in our original maze experiments.

> ![sbmrl-pic-10](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_10.png?raw=true)

---

作者的实验目的：通过增加每个任务分布中的任务的方差和长度，我们可以研究元强化学习算法的学习能力。

---

作者的实验结论：

1. 在最简单的任务中，除了BC+MAML以外，其他效果都很好。
2. 在中等难度任务中，作者的模型和 BC+PEARL 效果差不多。
3. 在最高难度任务中，只有作者的模型涨点起来了。

> ![sbmrl-pic-11](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_11.png?raw=true)

### LEARNING EFFICIENCY ON TARGET TASKS WITH FEW EPISODES OF EXPERIENCE

> ![sbmrl-pic-12](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_12.png?raw=true)
>
> <font color=blue size=2>还是保留我的观点，在迷宫环境，共享信息太容易获得，数据肯定有效。</font>

> ![sbmrl-pic-13](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_13.png?raw=true)
>
> <font color=blue size=2>直观上：1. 作者提出的方法和其他方法没有使用同样的平滑方法；2. 右边的Chicken Manipulation的方差未免也太大了吧......</font>

### C INVESTIGATING OFFLINE DATA VS. TARGET DOMAIN SHIFT

- 实验目的

  测试提出的方法是否适用于基于图像的观察，探究提出的方法对离线预训练数据和目标任务之间的领域转移的鲁棒性。

- 实验设置

  我们使用Pertsch等人的迷宫导航离线数据集，该数据集是在随机抽样的20×20迷宫布局上收集的，并在未见过的随机抽样的40×40测试迷宫布局上进行测试。

- 实验基线：自己的模型 + SPiRL

- 实验结论

  SiMPL可以通过将从离线数据集中学到的技能与高效的元训练相结合，更快地学习目标任务。

  这表明我们的方法可以扩展到基于图像的输入，并且对于离线预训练数据和目标任务之间的实质性领域转移具有鲁棒性。

> ![sbmrl-pic-4](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/%E7%A6%BB%E7%BA%BF+%E5%85%83%E5%BC%BA%E5%8C%96+%E6%8A%80%E8%83%BD%E6%8F%90%E5%8F%96%E5%92%8C%E4%BD%BF%E7%94%A8_14.png?raw=true)

### D EXTENDED RELATED WORK

#### Pre-training in Meta-learning

利用预训练模型来改进元学习方法已经在少样本图像分类做了探索。

我们可以将我们提出的框架视为具有预训练阶段的元强化学习方法。

具体而言，在预训练阶段，我们建议以自监督的方式从没有奖励或任务注释的离线数据集中首先提取可重用的技能和技能先验。然后，我们提出的方法从一组元训练任务中进行元学习，这显著加速了对未见过的目标任务的学习。

### E IMPLEMENTATION DETAILS ON OUR METHOD

- MODEL ARCHITECTURE

  Please refer to Pertsch et al. (2020) for more details on the architectures for learning skills and skill priors from offline datasets.

  > Karl Pertsch, Youngwoon Lee, and Joseph J. Lim. Accelerating reinforcement learning with learned skill priors. In Conference on Robot Learning, 2020.

- we adopt Set Transformer that consists of layers [2 × ISAB32 → PMA1 → 3 × MLP] for expressive and efficient set encoding.

  The output of the encoder is $(\mu_e, \sigma_e)$ which are the parameters of Gaussian task posterior $p(e|c) = N (e; \mu_e, \sigma_e)$. We varied task vector dimension dim($e$) depends on task distribution complexity. dim($e$) = 10 for Kitchen Manipulation, dim($e$) = 6 for Maze Navigation with 40 meta-training tasks, and dim($e$) = 5, otherwise.

- We employed 4-layer MLPs with 256 hidden units for Maze Navigation, and 6-layer MLPs with 128 hidden unit for Kitchen Manipulation experiment.

- We employ double Q networks to mitigate Q-value overestimation.

### F.2 PEARL AND PEARL-FT

PEARL 从元训练任务中学习，但不使用离线数据集。因此，我们直接在元训练任务上训练PEARL模型，而不需要从离线数据集中进行学习。我们使用从20个随机抽样的任务中平均梯度，其中每个任务梯度是通过从每个任务缓冲区中批量抽样计算得出的。目标熵设置为 $H = -dim(A)$， $α$ 初始化为0.1。

虽然Rakelly等人（2019）提出的方法不会在目标/元测试任务上进行微调，但我们将PEARL扩展为在目标任务上进行微调，以进行公平比较，称为PEARL-ft。由于PEARL不使用学习到的技能或技能先验，因此PEARL的目标任务学习仅是使用编码任务的初始化运行SAC。与我们的方法的目标任务学习类似，我们将Q函数和熵系数α初始化为元训练阶段学习的值。此外，我们在从任务无条件策略回滚中观察20个经验剧集后，将策略初始化为任务条件策略。用于微调的超参数与SAC相同。



