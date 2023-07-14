# 【论文笔记】Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning

更多笔记（在耕）：[这里](https://github.com/JinbiaoZhu/PaperReading)

本文已开源！[这里](https://guansuns.github.io/pages/llm-dm/)

---

这篇文章是我读的第一篇关于 LLM 的文章！

按照以下图片中的顺序来阅读的话，后面还会再读一篇 LLM 相关的~

![LLM世界模型规划_1](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/LLM%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E8%A7%84%E5%88%92_1.jpg?raw=true)

---

## Abstract

1. 研究背景？

   人们对将预训练大型语言模型（llm）应用于规划问题越来越感兴趣。

2. LLM 在使用时存在哪些不足？

   直接使用 LLM 作为规划器的方法目前是不切实际的，包括计划的**有限正确性**，强烈依赖于**模拟器与实际环境的交互**得到的反馈，以及利用**低效率的人类反馈**。

3. 作者的研究思路？

   > In this work, we introduce a novel alternative paradigm that constructs an explicit world (domain) model in planning domain definition language (PDDL) and then uses it to plan with sound domain-independent planners.

   在这项工作中，作者引入了一种新的替代范例，它使用**规划领域定义语言**（PDDL）构建显式的世界模型（领域模型），然后使用可靠的**领域无关规划器**进行规划。

4. 作者的具体技术路线？

   > To address the fact that LLMs may not generate a fully functional PDDL model initially, we employ LLMs as an interface between PDDL and sources of corrective feedback, such as PDDL validators and humans.

   为了解决LLMs可能最初无法生成全部功能的PDDL模型的问题，作者将LLMs作为**PDDL和纠正反馈来源**（例如**PDDL验证器**和**人类**）之间的**接口**。

   > For users who lack a background in PDDL, we show that LLMs can translate PDDL into natural language and effectively encode corrective feedback back to the underlying domain model.

   对于缺乏PDDL背景的用户，作者展示了**LLMs可以将PDDL翻译成自然语言**，并有效地将**纠正反馈**编码回到底层领域模型。

   > Our framework not only enjoys the correctness guarantee offered by the external planners but also reduces human involvement by allowing users to correct domain models at the beginning, rather than inspecting and correcting (through interactive prompting) every generated plan as in previous work.

   作者的框架不仅享受**外部规划器提供的正确性保证**，而且通过允许用户在开始时**对领域模型纠正**，而不是像以前的工作中那样**检查和纠正（通过交互提示）每个生成的计划**，从而减少了人类参与。

5. 实验设置？

   On two IPC domains and a Household domain that is more complicated than commonly used benchmarks such as ALFWorld

6. 实验指标？

   在摘要中暂无提到

7. 作者的结论是什么？

   We demonstrate that **GPT-4** can be leveraged to produce **high-quality PDDL models** for over **40 actions**, and the corrected PDDL models are then used to successfully **solve 48 challenging planning tasks**.

   关键：GPT-4、high-quality PDDL models、40 actions、solve 48 challenging planning tasks

8. 是否开源？

   有，[链接](https://guansuns.github.io/pages/llm-dm/)

## 1 Introduction

> LLMs have been tested to perform another widely-studied crucial aspect of AI agents, namely, sequential decision-making or planning. Preliminary studies suggest that, in some everyday domains, LLMs are capable of suggesting sensible action plans.

LLMs作为可执行AI智能体已被测试，并影响了另一个广泛研究的关键方向，即顺序决策或规划。初步研究表明，在某些日常领域中，LLMs能够提出<u>合理的行动计划</u>。

> LLM 怎么提出合理的计划并与现实相结合呢？作者引用了两篇文章。
>
> 《Language models as zero-shot planners: Extracting actionable knowledge for embodied agents》
>
> 这篇论文探讨了使用大型语言模型（LLMs）所学习的世界知识是否可以用于在交互式环境中行动。研究者们发现，如果预先训练的LLMs足够大并且得到恰当的提示，它们可以有效地将高级任务（例如“做早餐”）分解为可操作的中级计划（例如“打开冰箱”），而无需进一步的训练。然而，LLMs**天真地产生的计划**通常不能精确地映射到可接受的行动。为此，研究者们提出了一种程序，该程序基于现有的示范，并在语义上**将计划转换为可接受的行动**。在最近的 Virtual Home 环境中进行的评估表明，所得到的方法显著提高了可执行性。进行的人类评估显示了LLMs存在**可执行性**和**正确性**之间的权衡，但实验表明了**从语言模型中提取可操作知识**的有希望的迹象。
>
> 《Do as i can, not as i say: Grounding language in robotic affordance》
>
> 本文介绍了大型语言模型在机器人执行高级、时间延长的自然语言指令时，由于缺乏真实世界经验而难以应用的局限性。为了解决这个问题，作者提出了使用**预训练技能**来提供真实世界的基础，并将**低级技能**与**大型语言模型**相结合，以便让语言模型能提供**关于执行复杂程度和时间延长等方面的**指令的高级知识，同时这些技能的**价值函数**提供了**将这些知识与特定物理环境相连的**基础。作者在多个真实世界机器人任务中评估了该方法，证明了在需要真实世界基础上，这种方法能够完成长期、抽象、自然语言指令的移动机械手臂任务。

---

作者接下来提出大语言模型做规划**可能带来的问题**：

> However, the correctness and executability of these plans are often limited. 
>
> For instance, LLMs may regularly overlook the physical plausibility of actions in certain states and may not effectively handle long-term dependencies across multiple actions.

这些计划的正确性和可执行性通常存在问题。比如说，LLMs可能经常忽略某些状态下**动作的物理可行性**，并且可能无法有效地处理**跨多个动作的**长期的依赖关系。

---

针对这样的计划问题，作者提出了一种方法，并对这个方法做评价。

> One promising approach involves **collecting feedback from the environment** during plan execution and subsequently refining the plans. By incorporating various forms of feedback, such as **sensory information**, **human corrections**, or **information of unmet preconditions**, the planners can **re-plan** and **produce plans** that are closer to a satisficing plan.

在计划执行期间从环境中收集反馈，然后对计划进行优化。通过整合各种形式的反馈，例如<u>感官信息</u>、<u>人类更正</u>或<u>未满足前提条件的信息</u>，规划者可以<u>重新规划</u>并生成更接近满意计划的计划。

> Comments: emmmmmm收集反馈并根据反馈优化，感觉是一个很原始的思想啊；传统控制、深度学习的损失函数、强化学习的奖励函数也掺和一点，以及现在的反馈优化，想法相对朴素了。相比于人类更正和未满足前提条件信息，传感器的方法更贴近人感觉。。。

> - 传感器反馈：
>
>   《Inner monologue: Embodied reasoning through planning with language models》
>
>   这篇论文讨论了如何将LLMs的推理能力应用于**自然语言处理以外的领域**，如机器人的**规划**和**交互**。作者指出，在这些实体问题中，LLMs需要理解世界的许多语义，包括**可用的技能**、**这些技能如何影响世界**以及**世界变化如何反映到语言**中。作者研究了在实体环境中使用LLMs时，它们能否通过自然语言提供的反馈来进行推理，而无需额外的训练。作者认为，通过**利用环境反馈**，LLMs能够形成内心的自我对话，从而更丰富地处理和规划机器人控制场景。作者研究了各种反馈来源，如**成功检测**、**场景描述**和**人类交互**。研究结果表明，闭环语言反馈显著提高了三个领域的高级指令完成度，包括模拟和真实世界中的桌面重新排列任务以及厨房环境中的长期移动操作任务。
>
> - 人类更正反馈：
>
>   《React: Synergizing reasoning and acting in language models》
>
>   本文介绍了一种名为 React 的方法，它可以使 LLMs 在任务中同时**生成推理轨迹**和**特定于任务的动作**，从而实现更好的**协同作用**。<u>推理轨迹可以帮助模型诱导、跟踪和更新动作计划，并处理异常情况</u>，而动作则<u>允许模型与外部源进行交互，如知识库或环境，以收集额外的信息</u>。在多个语言和决策任务中，React的效果优于现有的方法，并且比没有推理或行动组件的方法具有更好的人类可解释性和可信度。
>
> - 未满足前提条件信息：
>
>   《Planning with large language models via corrective re-prompting》
>
>   提取LLMs中存在的常识知识，为设计智能的具身智能体提供了一条道路。相关工作已经使用了各种上下文信息，例如目标、传感器观测和场景描述，来查询LLMs以生成特定任务的高级行动计划；然而，这些方法通常涉及**人类干预**或**额外的机械装置**以实现从传感器到运动的交互。在这项工作中，作者提出了一种**基于提示**的策略，用于从LLM中提取可执行计划，该策略利用了一种**新颖且易于访问的信息源**：**前置条件错误**。方法假设在某些情况下**只能执行某些动作**，即必须**满足隐含的前置条件才能执行动作**（例如，必须解锁门才能打开它），并且具身智能体**具有确定在当前上下文中是否可以执行动作的能力**（例如，检测是否存在前置条件错误）。当智能体无法执行动作时，方法使用前置条件错误信息重新提示LLM，以提取可执行的纠正动作以在当前上下文中实现预期目标。在 Virtual Home 模拟环境中评估了方法，涉及<u>88个不同的任务和7个场景</u>。评估了不同的提示模板，并与从LLM中天真地重新采样动作的方法进行比较。方法使用前置条件错误可以改善计划的可执行性和语义正确性，并减少查询动作时需要重新提示的次数。
>
>   《Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents》
>
>   本文研究了在 Minecraft 中进行规划的问题，提出了两个主要挑战：（1）在像 Minecraft 这样的**开放式环境中**进行规划**需要精确和多步骤的推理**；（2）由于传统规划器不考虑当前智能体的接近程度，因此在复杂计划中**排序并行子目标**可能导致计划低效。为此，提出了一种基于 LLMs 的交互式规划方法“描述、解释、规划和选择”（DEPS），通过 `goalSelector` 模块对并行子目标进行排序来改进原始计划。实验结果表明，DEPS 方法可以有效地完成70多个 Minecraft 任务，性能接近翻倍。

---

作者接下来指出LLMs还是不能很好的成为规划器的原因：、

- LLMs在推理和规划方面尚未展现出足够的能力。

  最近的研究表明，即使提供了详细的**行动描述**，例如PDDL领域模型，或PDDL模型的自然语言版本，LLMs仍然难以生成正确和可执行的计划。

  > 《On the planning abilities of large language models (a critical investigation with a proposed benchmark)》
  >
  > 《PDDL planning with pre-trained large language models》

- 现有的 LLMs 规划范例仅允许以**完全在线的方式收集反馈**，这意味着反馈信号仅在智能体开始执行计划后才可用。然而，当好的模拟器使用成本高昂时，通过实际计划执行收集反馈**可能很昂贵**，并且可能无法充分利用**可证明安全规划的**优势。

- LLMs 表现出尚未完全理解的复杂行为，特别是在错误发生方面。LLMs 规划者**很容易在稍微不同的场景中重复相同的错误**。反复提供相同的反馈可能会让最终用户感到沮丧。

---

基于LLMs存在的不足，作者提出了什么？

1. 作者提出了一种基于模型的范例，**从LLMs中提取PDDL世界模型**，以**克服现有LLMs规划范式的限制**。

   该方法涉及**提供一组行动**及其**简要自然语言描述**的规划器，并指出初步研究表明，在某些日常领域中，LLMs能够提出合理的行动计划。

2. 作者的方法不是直接将用户命令映射到计划中，而是利用LLMs提取用PDDL操作模型的**行动符号表示**。

   这种中间输出可以与**外部领域无关的规划器**一起使用，可靠地搜索可行的计划，或者用于验证和纠正由LLMs规划器生成的“启发式”计划。

3. 作者的模块化方法本质上将规划过程分为两个不同的部分，即对**行动的因果依赖关系进行建模**和**确定完成目标所需的适当行动序列**。

4. 尽管LLMs可能无法在一开始生成无误的PDDL模型，但仍需考虑这一事实。

   为解决这个问题，LLMs还可以作为**PDDL和任何可提供自然语言纠正反馈的反馈来源**（例如人类和**VAL中的PDDL验证器**）之间的接口。LLM中间层将PDDL表示转换为自然语言，并呈现给用户进行检查。获取的反馈随后被合并并存档回PDDL模型。

   这将PDDL的复杂性隐藏在不具备PDDL先前知识的用户面前，并实现了无缝的反馈包含。

---

作者指出这样构造/结合的好处是：

- 通过利用LLMs将用户指令转换为PDDL中的目标规范，可以使用任何标准的**领域无关规划器**来搜索计划。

- 提取的PDDL模型可以用于**验证LLMs规划器建议的计划**，并提供未满足的前提条件，或目标条件的纠正反馈。在这种情况下，PDDL模型本质上充当**廉价高级模拟器**或**人类代理**，以确保计划的正确性。

  这减少了领域专家对忠实模拟器或计划的广泛手动检查的依赖。

- 与第一种方法相比，由于LLMs规划器的存在，第二种方法在将**显式和隐式用户约束**纳入常识领域方面可能提供更好的灵活性。

  > Compared to the first approach, the second approach potentially offers better flexibility in incorporating both explicit and implicit user constraints in common-sense domains because of the LLM planner.

但是作者又指出了：“However, as demonstrated in our experiments, although the validation feedback significantly improves the plan correctness on average, the performance of this approach is still limited by the "planning capability" of LLMs.”

> Comments: 我晕惹，作者的意思可能是，即使我提出了一种结合PDDL的规划方法，但是大语言模型自身的规划能力还是有限，所以有时候效果可能不那么好？

## 2 Related Work

作者在第二部分开始综述，这样可以更充分的事先了解一些大语言模型的工作。（反正比放在结论附近好......

### LLMs and planning.

现成的LLMs目前无法生成准确的计划。但是它们的生成的计划，可以用作“启发式方法”或“种子”，传递给外部规划器或人类操作者。

> SayCan and Text2Motion employ an LLM as a heuristic by utilizing it to score high-level actions, followed by a low-level planner that grounds these actions to determine the executability in the physical world.

SayCan 和 Text2Motion 使用LLM作为启发式方法，通过利用LLMs对高层次的动作进行评分，然后使用低层次的规划器将**这些动作**与**物理世界中的可执行性**相结合来确定它们是否可执行。

> 《Text2Motion: From Natural Language Instructions to Feasible Plans》
>
> 本文介绍了一种名为Text2Motion的语言规划框架，它能够帮助机器人解决需要长期推理的顺序操作任务。通过自然语言指令，该框架构建了一个任务和运动层面的计划，并验证 其能够达到 推断出的符号目标。Text2Motion使用Q函数库中的**可行性启发式编码**来指导任务规划，同时通过执行**几何可行性规划**来解决跨技能序列的几何依赖关系。作者在一系列**需要长期推理**、**抽象目标解释**和**处理部分可行性感知**的问题上对其进行了评估，结果表明Text2Motion的成功率为82%，而之前最先进的基于语言的规划方法仅达到13%。因此，Text2Motion为具有技能之间**几何依赖关系的语义多样的顺序操作任务**提供了有前途的泛化特性。

---

> In a similar vein, 2 works use LLMs to generate plans represented in Python-style code.

同样地，有两项工作使用LLMs生成以Python代码形式表示的计划。

> 《Code as policies: Language model programs for embodied control.》
>
> 这篇文章介绍了一种新的方法，利用基于自然语言的代码生成技术，将大型语言模型用于编写机器人策略代码。这些模型可以通过少量的示例命令和相应的策略代码，自主地编写新的策略代码，从而生成具有空间几何推理能力、适用于新指令、并且可以根据上下文（即行为常识）为模糊描述（例如“更快”）指定精确值（例如速度）的机器人策略代码。该方法已在多个真实机器人平台上进行了演示，并在 Human Eval 基准测试中取得了39.8%的解决问题的效果。
>
> 《Progprompt: Generating situated robot task plans using large language models.》
>
> 这段文字介绍了在机器人任务规划中使用大型语言模型的方法，以便评分潜在的下一步行动并生成行动序列。然而，现有方法要么需要枚举所有可能的下一步进行评分，要么生成可能在当前环境中不可行的自由形式文本。作者提出了一种编程式的语言模型提示结构，通过在环境中提供可用行动和对象的程序式说明以及可执行的示例程序来实现跨环境、机器人能力和任务的计划生成功能。作者通过实验证明了该方法在 Virtual Home 家庭任务中具有最先进的成功率，并在物理机械臂上部署了该方法。

---

> Other works have aimed to improve the planning performance of LLMs through prompt engineering or collecting various forms of feedback such as sensory information, human corrections, self-corrections or information of unmet preconditions.

其他工作旨在通过提示工程或收集各种形式的反馈，例如感官信息，人类更正，自我更正或未满足前提条件的信息，来改善LLMs的规划性能。

> 《LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models》
>
> 本研究旨在利用大型语言模型(LLMs)作为计划者，使具有视觉感知环境的体现智能体能够遵循自然语言指令完成复杂任务。现有方法的高数据成本和样本效率低，阻碍了开发多任务能力强、能够快速学习新任务的多功能体现智能体的发展。本文提出了一种新颖的方法LLM-Planner，利用大型语言模型进行少样本规划，进一步提出了一种简单而有效的方法来增强LLMs的物理基础，以生成和更新在当前环境中基于地面的计划。在ALFRED数据集上的实验表明，尽管使用不到0.5%的配对训练数据，LLM-Planner在少样本情况下也能取得非常有竞争力的性能：与使用完整训练数据训练的最近基线相比，LLM-Planner的性能相当。现有方法在同样的少样本设置下几乎无法成功完成任何任务。我们的工作为开发多功能和样本效率高的体现智能体快速学习多个任务打开了大门。
>
> 《Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling》
>
> 本文提出了一种方法，利用 LLMs 来假设抽象世界模型（Abstract World Model, AWM），并通过世界经验进行验证，以提高强化学习智能体的样本效率。具体而言，作者提出的 DECKARD 智能体在Minecraft中的物品制作任务中，分为两个阶段：（1）梦想阶段，智能体使用 LLMs 将任务分解为一系列子目标，即假设的AWM；（2）唤醒阶段，智能体为每个子目标学习一个模块化策略，并验证或纠正假设的AWM。这种使用LLMs假设AWM并根据智能体经验验证AWM的方法不仅使样本效率比当前方法提高一个数量级，而且对LLMs中的错误具有鲁棒性，并成功地将来自LLMs的嘈杂的互联网规模信息与基于环境动态的知识相结合。
>
> 《Reflexion: Language Agents with Verbal Reinforcement Learning》
>
> 该文介绍了 LLMs 作为目标驱动智能体与外部环境（如游戏、编译器、API）进行交互的趋势。然而，传统的强化学习方法需要大量的训练样本和昂贵的模型微调，使得这些语言智能体很难快速高效地从试错中学习。作者提出了一个新框架 Reflexion，通过语言反馈来加强语言智能体的决策能力，而不是通过更新权重。Reflexion 可以灵活地结合不同类型（标量值或自由形式的语言）和来源（外部或内部模拟）的反馈信号，并在不同任务（顺序决策、编码、语言推理）中实现显著的改进。作者还进行了消融和分析研究，探讨了不同反馈信号、反馈整合方法和代理类型对性能的影响。其中，Reflexion 在 Human Eval 编码基准测试中取得了 91% 的 pass@1 准确率，超过了之前的最先进模型 GPT-4 的 80%。

### Training transformers for sequential decision-making tasks.

> Along with using off-the-shell LLMs, there are works that either fine-tune LLMs or train sequence models for sequential decision making tasks.

除了使用现成的LLMs外，还有一些工作要么微调LLMs，要么训练序列模型用于顺序决策任务。

> 《 Plansformer: Generating symbolic plans using transformers》
>
> 这篇文章主要探讨了大型语言模型（LLMs）在自然语言处理（NLP）领域的应用以及其在自动化规划中的潜力。LLMs已经在问答、摘要和文本生成等自然语言任务中取得了超越最先进结果的成果。然而，将LLMs的文本能力扩展到符号推理方面的研究进展较慢，主要集中在解决与数学领域相关的问题。本文介绍了一种名为Plansformer的LLM，它经过细调用于规划问题，并能够生成具有正确性和长度优势的计划，而无需进行大量的知识工程。通过LLMs的迁移学习能力，Plansformer可以适应不同复杂度的规划领域。作者还通过在解决汉诺塔问题上的实验表明，Plansformer的一个配置可以实现约97%的有效计划，其中约95%是最优解。这篇文章为将LLMs应用于自动化规划领域提供了有益的探索和实践基础。
>
> 《Discovering underlying plans based on shallow models.》
>
> 本文介绍了计划识别的方法，旨在发现观察到的行动背后的目标计划。以前的方法要么通过将观察到的行动最大程度地“匹配”计划库来发现计划，假设目标计划来自计划库，要么通过执行行动模型来推断计划以最好地解释观察到的行动，假设完整的行动模型可用。然而，在现实世界的应用中，目标计划通常不来自计划库，并且完整的行动模型通常不可用。因此，本文提出了两种方法（DUP和RNNPlanner），基于行动向量表示发现目标计划。实验表明，这些方法能够在不需要提供行动模型的情况下发现不来自计划库的潜在计划，并且比传统的计划识别方法更有效。
>
> 《Pre-trained language models for interactive decision-making.》
>
> 这段文字提出了一种利用LLMs来支持学习和泛化的方法。通过将目标和观察表示为嵌入序列，并使用预训练的LLMs初始化策略网络，可以预测下一步的动作。实验证明，这种方法可以在不同环境和监督模式下实现有效的组合泛化。通过使用专家示范进行初始化并通过行为克隆进行微调，任务完成率在 Virtual Home 环境中提高了43.6%。然后，通过引入主动数据收集过程，智能体与环境进行交互，并使用新目标重新标记过去的“失败”经验，并在自我监督循环中更新策略。主动数据收集进一步提高了组合泛化能力，优于最佳基准模型25.1%。最后，通过研究语言模型策略的三个可能因素来解释这些结果。发现序列输入表示和基于LM的权重初始化对泛化都很重要。然而，策略输入编码的格式（例如自然语言字符串还是任意序列编码）几乎没有影响。总之，这些结果表明，语言建模产生的表示不仅对语言建模有用，而且对于建模目标和计划也很有用；这些表示即使在语言处理之外，也可以帮助学习和泛化。
>
> 《Planning with sequence models through iterative energy minimization》
>
> 这段文字讨论了如何将序列建模与规划相结合，以提高强化学习策略的性能。作者提出了一种基于迭代能量最小化的方法，通过训练一个掩码语言模型来捕捉行动轨迹的隐式能量函数，并将规划问题转化为寻找能量最小的行动轨迹。实验证明，这种方法在 BabyAI 和 Atari 环境中能够改善强化学习的性能，并具有新的任务泛化、测试时间约束适应和计划组合等独特优势。

在这项工作中，作者使用现成的LLM构建符号世界模型，而无需进行任何额外的训练。

### Learning/acquiring symbolic domain models.

在经典规划中，已经探索了许多基于学习的方法和基于交互式编辑器的方法来获取符号域模型。

作者提到了在构建领域模型时，利用LLMs中嵌入的常见世界知识和它们的上下文学习能力的重要性。

最近的研究表明，LLMs在将自然语言翻译为形式描述或从自然语言指令中构建PDDL目标方面非常有效。

## 3 Problem Setting and Background

作者的工作重点是智能体从用户中接收到高级指令或任务 $i$ 的情境。智能体只能执行技能库 $\Pi$ 一部分的技能或操作，其中每个技能 $k$ 都有一个简短的语言描述 $l_{k}$ 。假设智能体已经配备了与这些高级技能相对应的低级控制策略。为了实现 $i$ 中指定的目标条件，规划器（可以是**LLMs**或**外部规划器**）需要提出**智能体可以执行的一系列高级技能序列**。

### 3.1 Classical planning problems

经典规划问题可以用元组 $P = ⟨D,I,G⟩$ 形式化表示。$D(omain)$ 称为领域，$I(nit)$ 是初始状态，$G(oal)$ 是目标。**规划问题的状态空间由谓词的真值分配组成**。

领域 $D$ 由元组 $D = ⟨F,A⟩$ 进一步定义。$F(luent)$ 对应于流变量的集合，即用于定义状态空间的状态变量，**每个流变量对应于具有某些参数的谓词**。 $A(ctions)$ 对应于可以执行的操作集合。每个操作 $a^{i}[V] \in A$ （其中 $V$ 是运算符 $a^{i}$ 使用的变量集，每个变量可以映射到一个对象）可以通过两个组件进一步定义，即描述操作**何时可以执行的前置条件** $\text{prec}[V]$ 和定义**操作执行时会发生什么的效果** $\text{eff}[V]$ 。

假设 $\text{prec}[V]$ 由在变量 $V$ 上定义的一组谓词组成。只有当操作的前置条件满足时，即前置条件中的谓词在给定状态下成立，操作才可执行。效果集 $\text{eff}[V]$ 由元组 $⟨\text{add}[V],\text{del}[V]⟩$ 进一步定义，其中 $\text{add}[V]$ 是操作将设置为真的谓词集合， $\text{del}[V]$ 是操作将设置为假的谓词集合。如果用一个对象替换每个变量，则称这样的操作为 `grounded`，否则称为抽象的操作模型。

解决规划问题的方案称为计划，它是一系列动作，一旦在初始状态中执行，就会导致一个状态，其中目标规范成立。经典规划问题是规划中较简单的类之一，有多个扩展，支持更复杂的前置条件、条件效果和更丰富的规划形式。

### 3.2 PDDL

Planning Definition and Domain Language (PDDL) 是经典规划问题的标准编码语言。

![LLM世界模型规划_2](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/LLM%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E8%A7%84%E5%88%92_2.png?raw=true)

`?x` 表示要放下方块的动作，也就是对应上面的 `PutDownBlock` 。前置条件说明机器人必须用夹爪拿着方块。效果行描述了这个动作的预期结果。

## 4 Methodology

![LLM世界模型规划_3](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/LLM%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E8%A7%84%E5%88%92_3.png?raw=true)

PDDL提供了一种简洁和标准化的方式来表示世界模型。一旦构建了PDDL模型，任何领域无关的规划器都可以无缝地使用它来搜索计划，给定初始状态和目标条件。在本节中，我们将介绍使用LLMs构建PDDL模型的解决方案。并讨论纠正生成的PDDL模型中错误的技术。最后，展示使用生成的PDDL模型解决规划问题的完整流程。

### 4.1 Constructing PDDL models with LLMs

对预训练大语言模型使用提示工程，具体使用以下：

1. Detailed instructions for the PDDL generation task, outlining components of upcoming inputs and desired outputs.

   详细说明PDDL生成任务的指令，概述即将输入的组件和期望的输出。

2. One or two examples from other domains (e.g., the classical `Blocksworld` domain) for illustrating the input and output formats.

   从其他领域（例如经典的 `Blocksworld` 领域）选择一个或两个示例，以说明输入和输出格式。

3. A description of the current domain, including contextual information about the agent’s tasks and physical constraints due to the specific embodiment of the agent.

   描述当前领域，包括有关智能体任务和物理约束的上下文信息，这些约束由智能体的特定实现方式引起。

4. A description of the agent’s action and a dynamically updated list of predicates that the LLM can reuse to maintain consistent use of symbols across multiple actions.

   对智能体动作的描述，以及动态更新的谓词列表，LLMs可以重复使用这些谓词来在多个动作之间保持符号的一致使用。

---

作者在论文中展示了一个提示词的样子：

![LLM世界模型规划_4](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/LLM%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E8%A7%84%E5%88%92_4.png?raw=true)

> Comments: 好累~先看到这里吧 2023.07.13......

根据行动描述或领域背景中包含的信息，用户可以获得不同程度的、对提取到的PDDL的控制，或者从LLMs获得不同程度的支持/帮助。

一方面，当用户只提供行动的最小描述，例如“此操作使机器人使用微波炉加热食物”时，不仅使用LLMs作为PDDL构造器，还利用模型中**编码的通用世界知识**进行知识获取。这样做有助于扩展AI智能体的动作集。

另一方面，当提示中明确提到某些先决条件或效果时，更依赖LLMs解析**自然语言中提供的知识**并**通过设计谓词的集合**来精确表示它的能力。当技能可能有不同的初始设置，并且工程师在设计技能时已经对先决条件做出了一些假设时，这种能力非常有用。

---

期望的输出包括以下元素：行动的参数列表；用PDDL表达的先决条件和效果；如果适用，任何新定义的谓词及其自然语言描述的列表。

任何新定义的谓词都将添加到一个积极维护的谓词列表中，以便LLMs可以在后续操作中重复使用现有的谓词，而不会创建冗余的谓词。一旦获得了初始的PDDL模型和完整的谓词列表，就会重复整个过程，但是将所有提取出来的谓词呈现给LLM。运行两次生成过程很有效果，因为LLMs在第一次迭代中**可能不知道某些先决条件**，特别是如果先决条件没有明确提到的话。

> One alternative to this action-by-action generation could be to include descriptions of all the actions in the prompt and require the LLM to construct the entire domain model in a single dialogue.

一个替代逐个动作生成的方法是，在提示中包含所有动作的描述，并要求LLMs在单个对话中构建整个领域模型。

---

值得注意的是，每次定义新谓词时，LLMs都需要给出自然语言描述。接下来章节会提到，这对于**使任何用户都能够轻松理解和检查**生成的PDDL模型；而**无需深入研究低级符号表示**是至关重要的。此外，自然语言描述允许使用LLMs将自然语言环境描述**翻译**为PDDL，或利用**预训练的视觉语言**模型，并以**问答方式查询它们**，基于环境的观察来自动地对初始状态的谓词值进行实例化。

### 4.2 Correcting errors in the initial PDDL models

与任何涉及LLMs的用例一样，无法保证输出完全没有错误。将纠错机制纳入其中至关重要。

虽然PDDL专家可以**直接检查**和**纠正生成的PDDL模型**，但不能假设所有终端用户都具备这种专业水平。

解决方案是将LLMs用作**底层PDDL模型**和**任何可以提供自然语言纠错反馈的反馈源**之间的**中间层或接口**。

作者在论文中考虑了两个反馈来源，即**PDDL模型验证工具**（例如VAL中的工具）和**人类领域专家**。前者用于检测基本语法错误，后者主要负责捕捉事实错误，例如缺少效果。值得注意的是，反馈来源不限于上述提到的，后续还有其他的反馈源。

对于来自PDDL验证器的纠正反馈，生成的PDDL模型直接呈现给验证器以获取简短但可读的错误消息。对于来自用户的纠正反馈，基于谓词和参数的自然语言描述将PDDL模型转换为其自然语言版本。用户可以检查潜在的错误动作模型。人类纠正可以**在构建PDDL模型期间**和**模型用于执行计划后**发生。

虽然有技术可用于帮助用户定位模型中的错误，但这超出了本工作的范围，因为这里的重点是调查**使用LLMs根据反馈纠正PDDL模型的可行性**。

纠正动作模型并不比纠正计划或LLM规划器的“推理轨迹”更具认知要求。实际上，在纠正计划时，人类还必须记住动作模型及其因果链以验证计划。一旦动作模型被纠正，用户就不再需要重复提供类似的反馈。最后，通过重播和继续PDDL构建对话来集成纠正反馈。

### 4.3 Generating plans with the extracted PDDL models

给定提取的谓词集合及其自然语言描述，可以使用LLMs将**环境描述**翻译为PDDL来获得实例化的初始状态，或通过**观察环境**并**查询预训练的视觉语言模型**来获得。此外，可以使用LLMs解析用户的命令并将其转换为符号形式来获取目标。通过这种设置，一种简单直接的方法是**使用标准的领域无关规划器**可靠地找到满足或甚至最优的计划来实现指定的目标。

在常识领域中，LLMs可能会生成有意义的“启发式”，LLMs计划也可以用作本地搜索规划器的种子计划，以加速计划搜索。还可以使用提取的PDDL作为符号模拟器或人类智能体，基于验证信息向LLMs规划器提供纠正反馈。通过这种设置，规划器可以**通过重新提示**迭代地改进计划。

根据具体的问题设置，提取的PDDL模型也可以用于任务规划以外的其他任务。

> For instance, in cases where reinforcement learning is permissible, the domain model can be used to guide skill learning [19, 7] or exploration even if the model is not fully situated [13].

例如，在允许强化学习的情况下，即使模型不完全显现/展示，领域模型也可以用于指导技能学习或探索。

> 《 Guided skill learning and abstraction for long-horizon manipulation》之前读的那一篇~
>
> 《Symbolic plans as high-level instructions for reinforcement learning》

## 5 Empirical Evaluation

本文在一个日常家庭机器人领域和两个更专业的IPC领域（即 Tyreworld 和 Logistics）上进行了实验。

首先评估LLMs生成的PDDL模型的质量，然后评估LLMs整合来自PDDL验证器和用户的纠正反馈以获得无错误PDDL模型的能力，并展示使用经过纠正的PDDL模型进行下游规划任务的多种方式。

还介绍了GPT-4和GPT-3.5-Turbo在PDDL构建方面的结果。

### 5.1 Constructing PDDL

在PDDL构建任务中，作者旨在研究LLMs在获得**领域专家的纠正反馈之前**能够构建准确的PDDL模型的程度。对于所有领域，来自经典 `Blocksworld` 领域的两个动作被用作提示中的demo，以便终端用户不需要提供任何特定于领域的示例。

为了评估正确性程度，作者团队招募了多名具有PDDL专业知识的研究生，他们将扮演专家。这些专家负责注释和纠正生成的PDDL模型中存在的任何错误。作为评估指标，作者计算和报告注释的总数，这些注释可能**包括删除不相关的前提条件**、**添加缺失的前提条件**、**替换不正确的谓词**、**包含缺失的参数**和**其他常见的更正**。

> Comments: 根据论文，作者招募专家其实是对生成的 PDDL 模型进行增删改查的纠正操作。

请注意，注释数量可以视为生成的PDDL模型及其纠正版本之间的近似距离/差距。

为了全面了解生成模型的质量，还在附录中列出了所有模型和收集的注释。在每个图中，影响PDDL模型功能的错误用黄色突出显示，而次要问题用绿色突出显示。次要问题的一个例子是在前提条件中冗余地包含（`pickupable ?o`），当（`robot-holding ?o`）已经列出时。前者是不必要的，因为它可以由后者暗示，但这只影响简洁性而不是功能。

---

![LLM世界模型规划_5](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/LLM%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E8%A7%84%E5%88%92_5.png?raw=true)

1. 首先评估当给定部分约束信息时生成的PDDL模型，因为这更接近于大多数实际使用情况，其中对技能库 $\pi$ 中的技能的约束通常是预先指定的。在这种设置下，评估重点是LLMs准确恢复捕获所述约束和技能之间潜在依赖关系的“基础真实PDDL”的能力。

   与 `GPT-3.5-Turbo` 相比， `GPT-4` 可以生成具有显着更少错误的高质量PDDL模型。在 `GPT-4` 产生的59个错误中，其中三个是语法错误，其余是事实错误，如缺少前提条件和效果。

   这一观察结果表明，虽然 `GPT-4` 表现出遵循PDDL语法的熟练程度，但它可能仍然对动作有不准确的理解。

2. 通过检查谓词集合，还发现 `GPT-4` 可以设计一组直观命名的谓词，可以简洁而精确地描述域中对象和事件的状态。相比之下， `GPT-3.5-Turbo` 产生了高度嘈杂的输出，有超过350个错误。

   这表明我们的框架在很大程度上依赖于 `GPT-4` 在理解符号方面的改进能力，未来的工作可能会调查**如何启用更轻量级的模型的使用**（例如，通过对一些PDDL数据集进行微调）。

---

当动作描述包含最少信息时，LLMs也可以用于提出前提条件和效果以协助知识获取。为了验证这个假设，在可以具有更开放式动作设计的家庭领域上进行了额外的实验。在这种设置下，动作模型的正确性基于前提条件和效果是否在动作之间建立了正确的连接来确定。

`GPT-4` 可以提出有意义的动作模型，并且生成的PDDL模型只有大约45个错误。

---

尽管 `GPT-4` 在PDDL构建任务中表现出了改进的性能，但我们的实验仍然揭示了一些限制。

- `GPT-4` 在行动之间的因果关系方面仍然**表现出浅显的理解**，特别是涉及到像空间推理这样的推理技能的任务。在构建“从家具上拿起物体”的动作模型时，即使提供了相关谓词（这些谓词是在“堆叠物体”动作中创建的），GPT-4仍未考虑到**可能有其他物体堆叠在目标物体上**。
- `GPT-4` 可能会输出矛盾的效果。例如，在用搅拌机捣碎食物的动作中， `GPT-4` 同时列出了(`not (object-in-receptacle ...)`)和(`object-in-receptacle ...`)作为效果。

### 5.2 Correcting PDDL with domain experts

这部分的目标是展示使用 `GPT-4` 作为中间层将**自然语言反馈**纳入并纠正PDDL模型的可行性。

使用PDDL验证器捕获基本语法错误。在家庭领域中，由于参数的对象类型问题，与相关谓词的不当使用相关联的有两个语法错误。例如，通过继续使用PDDL构造对话框与反馈信息 `object-on` 的第二个参数应该是`furnitureAppliance` ，但给出了 `householdObject` ，`GPT-4` 可以定位不准确的PDDL代码片段并将其替换为正确的代码。

对于其他事实错误， `GPT-4` 基于自然语言反馈成功纠正了所有错误。一个关于事实错误的反馈信息示例是“缺少效果：在捣碎后，物品不再可拾取。”

作者还尝试使用各种方式编写的反馈，并且 `GPT-4` 能够理解所有消息并成功纠正模型。为了量化 `GPT-4` 有效利用领域专家反馈的程度，计算了有关事实错误的**反馈消息数量**。我们的结果表明， **GPT-4** 需要59个反馈消息来解决总共56个事实错误。

有三种情况需要额外的反馈。其中一种情况涉及用户重申错误，而另外两种情况涉及 `GPT-4` 引入新错误。此外，我们尝试使用 `GPT-3.5-Turbo` 纠正相同的错误。结果表明， `GPT-3.5-Turbo` 不仅无法纠正所有错误，而且偶尔会引入新错误，再次证实其无法操作符号的能力不足。

### 5.3 Generating plans with the extracted PDDL models

对于规划任务（即用户指令和初始状态），使用家庭领域和物流领域，其中最先进的LLMs规划器很难找到有效的计划。对家庭领域采样了27个任务，对物流领域采样了21个任务。对于初始状态，提供了基本假设，并且对于目标，利用 `GPT-4` 将用户指令翻译成基于提取的谓词的PDDL目标规范，并将其发送到已经通过LLMs获得领域模型的标准STRIPS规划器。通过这种设置，经典规划器 Fast Downward 可以在95%的情况下有效地找到有效的计划。失败仅由于目标翻译错误，表明通过LLMs获得的领域模型非常稳健。

请注意，与早期的方法仅将LLMs用作将用户目标转换为PDDL格式的机制，并将其发送到具有手工制作的正确PDDL领域模型的外部可靠规划器相比，本文方法使用LLMs本身来开发驱动外部规划器的PDDL世界模型。

---

对于利用PDDL模型验证LLMs规划的方法，采用了最先进的算法 React，并将 `GPT-4` 作为潜在的LLM规划器。

但是，我们对提示设计进行了两个修改。

1. 提供了所有动作的详细描述，包括参数、前提和效果。

   这些描述是通过使用另一个LLMs将生成的PDDL领域模型转换为自然语言获得的。

2. 仅使用每个领域的两个固定示例，因为最终用户可能无法始终提供大量示例，并且规划器应依赖动作模型信息。LLM计划、符号目标规范、初始状态和领域模型传递给计划验证系统（即VAL）以检查未满足的前提条件或目标条件。

然后，将验证结果（以PDDL形式给出）使用 `GPT-4` 转换为自然语言，并通过继续规划对话提供给LLM规划器。在我们的实验中，由于对 `GPT-4` 的访问受限，将每个任务的反馈次数限制为8次。

![LLM世界模型规划_6](https://github.com/JinbiaoZhu/PaperReading/blob/main/pic/LLM%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E8%A7%84%E5%88%92_6.png?raw=true)

普通的LLMs规划器经常**忽略动作前提条件**，并且成功率极低。通过集成验证反馈，观察到计划正确性有了显着改善。尽管有此改进，总体性能仍然不令人满意，因为成功率仍然低于50％。观察到 `GPT-4` 无法有效利用反馈，经常陷入循环，反复生成相同的计划。在某些情况下，它还可能在试图纠正计划时引入新错误。

---

除了正确性概念外，实验还揭示了LLM规划器的有趣特性。在家庭领域中，有意在某些指令中引入无法使用现有谓词表达的排序约束。值得注意的是，通过手动检查生成的计划，观察到所有LLMs计划**都遵循指定的顺序**，尽管**不是完全正确或可执行**。

此外，在家庭领域中，还观察到**经典规划器**偶尔会**生成物理上可行**但**不寻常的动作**，例如在不使用刀具时将刀具放在烤面包机上。相比之下，LLMs规划器很少展示这样的动作，**表明LLMs具有隐含的人类偏好知识**。具有意义的做法是，探索更有效地结合**LLMs规划器的优势**和**符号领域模型**提供的正确性保证的方法，特别是确定应保留LLM计划中哪些信息。

## 6 Conclusion

> Apart from directions for further research that we have previously mentioned, there are several exciting opportunities for extending this work. Firstly, the complexity of our evaluation domains is still lower than that of many domains used in the classical planning literature. It remains to be seen whether LLMs can effectively scale to write PDDL models that express more intricate logic. Secondly, our framework assumes full observability, meaning that the agent must fully explore the environment to acquire object states at the beginning. It would be useful to support partial observability. Finally, our experiments assume the grounding of predicate values is done perfectly. However, it would be useful to take into account that perception can be noisy in practice.

除了我们之前提到的进一步研究方向之外，还有几个扩展此工作的令人兴奋的机会。

1. 评估领域的复杂性仍然低于经典规划文献中使用的许多领域。

   尚不清楚LLMs能否有效扩展以编写表达更复杂逻辑的PDDL模型。

2. 框架假定完全可观察性，这意味着智能体必须完全探索环境以在开始时获取对象状态。

   支持部分可观察性将非常有用。

3. 实验假定谓词值的基础是完美的。然而，考虑到实践中感知可能存在噪声。

# Appendix

## A. 1 Broader impact on using LLMs

人们普遍倾向于将LLMs用于各种任务，包括计划生成。

考虑到LLMs不能保证生成正确的计划，这可能会导致下游的安全问题。

从LLMs中提取领域模型，并将其与外部可靠的规划器结合使用，以减轻这些安全问题。

鉴于人类仍然负责验证从LLMs中提取的领域模型的正确性，仍有可能**无意中认证不正确**或**不理想的领域模型是正确的**，从而导致不良计划和智能体行为。

## A. 2 Additional discussion on alternative to action-by-action PDDL construction

作者的 action-by-action 生成的一种替代方法是在提示中包含所有操作的描述，并要求LLMs在单个对话中构建整个领域模型。这种方法可能使LLMs能够更好地建立**所有操作的全局视图**。

但是，作者在实际中并不采用这种替代方案，原因如下：

- 包含所有操作可能导致提示过长，可能超过LLMs的上下文窗口大小（`max_tokens`）。

  这可能会对使用较小的语言模型（例如 `GPT-3.5-Turbo`）或尝试训练较小的专用模型造成实际问题；

- 纠正性反馈的整合依赖于**继续构建对话**，这需要一个**较短的初始提示**以适应上下文窗口；

实验表明， action-by-action 行动构建方法已经取得了令人满意的结果。

## A. 3 Examples of feedback messages that capture syntax errors

作者用VAL中的PDDL验证器来识别语法错误。但是，使用简单的Python脚本就可以轻松检测到几个“较简单”的语法错误。作者在实验中编写了脚本来捕获此类语法错误。

请注意，由于可以以最小的成本检测到这些错误，因此相应的反馈消息直接提供给LLMs。

1. 在这项工作中，我们仅考虑标准的基本层PDDL。但是，LLMs可能已经看到了各种PDDL扩展，并可能在构建的领域模型中使用它们。因此，**在检测到不支持的关键字**时向LLM提供反馈消息。

   一个示例反馈是：“前提条件或效果包含关键字 `‘forall’` ，这在标准STRIPS风格模型中不被支持。请用简化的方式表达相同的逻辑。如果需要，您可以提出新的谓词（但请注意，应尽可能使用现有的谓词）。”

2. 新创建的谓词可能与**现有的对象类型具有相同的名称**，这在PDDL中是不允许的。在这种情况下，将向LLM供反馈消息以通知其名称冲突。

   一条消息可能会说明：“以下谓词与现有的对象类型具有相同的名称：1. `‘smallReceptacle’`。请重新命名这些谓词。”

3. **新创建的谓词可能与现有的谓词具有相同的名称**，这在PDDL中是不允许的。此外，LLMs经常错误地将**现有谓词**列在“**新谓词**”部分下。在这种情况下，**将向LLMs提供反馈消息以通知其名称冲突或错误**。

   一条消息可能会说明：“以下谓词与现有谓词具有相同的名称：1. `(cutting-board ?z - smallReceptacle)` ，如果小容器 `?z` 是切菜板，则为真。您应尽可能重用现有谓词。如果您正在重用现有的谓词，您不应将它们列在“新谓词”下。如果现有谓词不足而您正在设计新谓词，请使用与现有谓词不同的名称。请修改PDDL模型以修复此错误。”这是我们在实验中发现GPT-4最常见的语法错误。

4. LLMs可能会未能仅使用提示中给出的对象类型。

   一个示例反馈可以是：“参数 `?p` 的对象类型 `‘pump’` 无效。”

5. 在一些不太常见的情况下， `GPT-4` 可能会出现谓词使用问题，通常是由于**对象类型不匹配**引起的。这种类型的错误可以被VAL捕获。

   一个示例反馈消息可以是：“存在语法错误，`‘object-on’` 的第二个参数应该是 `furnitureAppliance` ，但给出了 `householdObject` 。如果需要，请使用正确的谓词或设计新的谓词。”

## A. 4 Techniques that assist users to locate errors in PDDL models

有几种已经成熟的技术和工具可用于定位PDDL模型中的错误。

GIPO 等图形工具可以有效地可视化动作之间的因果依赖关系。然而，这些高级工具或技术超出了本文的范围。

在这里，作者概述了一个可行的解决方案，作为对那些不熟悉这些工具的用户的起点。

在纠正反馈方面，有两个阶段可以获得：在**构建PDDL模型期间**以及**在使用域模型生成计划时**。

1. 最终用户可以直接查看域模型并识别潜在的事实错误。由于所有谓词和参数都附带有自然语言描述，可以轻松地将PDDL模型转换为自然语言并呈现给用户。这使用户能够预先筛选前提条件和效果。请注意，在此阶段捕获不希望所有实际错误，因为用户可能直到查看最终计划之前才意识到某些限制。

2. PDDL模型用于解决下游规划问题。这里可能会出现两种可能的情况：对于给定的目标规范，找不到计划，或者至少找到一个计划，但它被用户拒绝了或在实际环境中导致执行失败。

   为了解决第一种情况，可以要求用户建议一个目标满足计划，该计划应该是可执行的（但不一定是最优的）。然后，使用生成的PDDL模型来“验证”建议的计划。这使得能够找到计划中第一个具有不满足前提条件的步骤。然后将此步骤之前所有操作的模型以及未满足的前提条件转换为自然语言并呈现给用户进行检查。

   在 `GPT-4` 提取的“切割物体”模型中，模型要求物体同时放置在切菜板和家具上，这在物理上是不可能的。通过利用用户建议的计划，可以确定可能存在错误的模型，并标记不正确的前提条件。在第二种情况下，如果PDDL模型提供了无效的计划，则通常在执行失败期间和之前的操作中缺少前提条件或效果。用户可以关注这些操作。

## A. 5 Detailed description of the Household domain

考虑一个单臂机器人模型，它与SPOT机器人和Fetch机器人非常相似。

机器人无法同时抓取多个物体或在持有不相关物品时执行操作（例如，在持有杯子时打开冰箱门）。

确保约束与现实世界的机器人能力相一致。例如，机器人手臂可能比人类手臂更不灵活，因此，要求在具有开放和灵活表面的家具上执行某些操作（例如，当午餐盒放在厨房台面上而不是冰箱里时，机器人只能从午餐盒中取出食品）。此领域的PDDL构造提示包括领域的一般描述，概述机器人要执行的任务、涉及的物体类型以及机器人形态的详细信息。

## A. 6 Household: Constructing PDDL Models

这里的AI智能体是一个家庭机器人，可以导航到房子里的**各种大型**和**通常不可移动的家具或电器**，**执行家务任务**。

请注意，机器人只有一个夹爪，因此它只能做以下工作：

1. 拿着一个物体；
2. 在执行某些操作（例如打开抽屉或关闭窗户）时，不应在夹爪中拿着任何其他不相关的物品；
3. 对小型家庭用品的操作应在具有平坦表面的家具上进行，以获得足够的操作空间。

此领域中有三种主要类型的对象：机器人、家具电器和家庭用品。

对象类型 `furnitureAppliance` 涵盖了大型和通常不可移动的家具或电器，例如炉灶燃烧器、边桌、餐桌、抽屉、橱柜或微波炉。对象类型 `householdObject` 涵盖所有其他小型家庭用品，例如手持式吸尘器、布、苹果、香蕉和小容器（如碗和午餐盒）。其中有一个名为 `smallReceptacle` 的 `householdObject` 子类型，涵盖小容器，例如碗、午餐盒、盘子等。

在此领域中，机器人和小型家庭用品（例如苹果、橙子、碗、午餐盒或灯）的位置由大型和通常不可移动的家具或电器确定。

接下来作者展示了很多的关于动作的提示词：这里举例一些~

### A. 6. 1 An example prompt for constructing PDDL models of the action "close a small receptacle"

| An example prompt for constructing PDDL models of the action "close a small receptacle" |
| ------------------------------------------------------------ |
| You are defining the preconditions and effects (represented in PDDL format) of an AI agent's actions. Information about the AI agent will be provided in the domain description. Note that individual conditions in preconditions and effects should be listed separately. For example, “object_1 is washed and heated” should be considered as two separate conditions “object_1 is washed” and “object_1 is heated”. Also, in PDDL, two predicates cannot have the same name even if they have different parameters. Each predicate in PDDL must have a unique name, and its parameters must be explicitly defined in the predicate definition. It is recommended to define predicate names in an intuitive and readable way.</br>One or two examples from other domains for illustrating the input and output formats<br/>Here are two examples from the classical BlocksWorld domain for demonstrating the output format.<br/>Domain information: BlocksWorld is a planning domain in artificial intelligence. The AI agent here is a mechanical robot arm that can pick and place the blocks. Only one block may be moved at a time: it may either be placed on the table or placed atop another block. Because of this, any blocks that are, at a given time, under another block cannot be moved. There is only one type of object in this domain, and that is the block.<br/>Example 1<br/>Action: This action enables the robot to put a block onto the table. For example, the robot puts block_1 onto the table.<br/>You can create and define new predicates, but you may also reuse the following predicates: No predicate has been defined yet.<br/>Parameters:<br/>1. ?x - block: the block to put down<br/>Preconditions:<br/>```(and(robot-holding ?x))```<br/>Effects:<br/>```(and(not (robot-holding ?x))(block-clear ?x)(robot-hand-empty)(block-on-table ?x))```<br/>New Predicates:<br/>1. (robot-holding ?x - block): true if the robot arm is holding the block ?x<br/>2. (block-clear ?x - block): true if the block ?x is not under any another block<br/>3. (robot-hand-empty): true if the robot arm is not holding any block<br/>4. (block-on-table ?x - block): true if the block ?x is placed on the table<br/>Example 2<br/>Action: This action enables the robot to pick up a block on the table.<br/>You can create and define new predicates, but you may also reuse the following predicates:<br/>1. (robot-holding ?x - block): true if the robot arm is holding the block ?x<br/>2. (block-clear ?x - block): true if the block ?x is not under any another block<br/>3. (robot-hand-empty): true if the robot arm is not holding any block<br/>4. (block-on-table ?x - block): true if the block ?x is placed on the table<br/>Parameters:<br/>1. ?x - block: the block to pick up<br/>Preconditions:<br/>```(and(block-clear ?x)(block-on-table ?x)(robot-hand-empty))```<br/>Effects:<br/>```(and(not (block-on-table ?x))(not (block-clear ?x))(not (robot-hand-empty))(robot-holding ?x))```<br/>New Predicates:<br/>No newly defined predicate |
| Here is the task.<br/>A natural language description of the domain<br/>Domain information: The AI agent here is a household robot that can navigate to various large and normally immovable furniture pieces or appliances in the house to carry out household tasks.<br/>Note that the robot has only one gripper, so (a) it can only hold one object; (b) it shouldn't hold any other irrelevant objects in its gripper while performing some manipulation tasks (e.g., opening a drawer or closing a window); (c) operations on small household items should be carried out on furniture with a flat surface to get enough space for manipulation. There are three types of objects in this domain: robot, furnitureAppliance, and householdObject. The object type furnitureAppliance covers large and normally immovable furniture pieces or appliances, such as stove burners, side tables, dining tables, drawer, cabinets, or microwaves. The object type householdObject covers all other small household items, such as handheld vacuum cleaners, cloth, apples, bananas, and small receptacles like bowls and lunch boxes. In this domain, the locations of the robot and small household items (e.g., apples, oranges, bowls, lunch boxes or lamps) are determined by large and normally immovable furniture pieces or appliances.<br/>A natural language description of the action<br/>Action: This action enables the robot to close a small receptacle receptacle_1 that is openable (e.g. small storage boxes or lunch boxes with lids). For example, the robot closes lunch_box_2, or the robot closes storage_box_1.<br/>The dynamically updated list of predicates<br/>You can create and define new predicates, but you may also reuse the following predicates:<br/>1. (robot-at ?x - furnitureAppliance): true if the robot is at the furniture or appliance ?x<br/>2. (object-on ?x - householdObject ?y - furnitureAppliance): true if the object ?x is on or in the<br/>furniture or appliance ?y , →<br/>3. (pickupable ?x - householdObject): true if the object ?x can be picked up by the robot<br/>4. (closed-receptacle ?x - smallReceptacle): true if the small receptacle ?x is closed |

## A. 8 Logistics: Constructing PDDL Models

这里的AI智能体是一个物流规划师，必须计划通过卡车在城市内部和通过飞机在城市之间运输包裹。

在城市内部，位置直接相连，允许卡车在任何两个位置之间行驶。同样，城市之间也直接相连，允许飞机在任何两个城市之间飞行。每个城市都配备有一辆卡车，并有一个指定的位置作为机场。共有五种类型的对象：包裹、卡车、飞机、位置和城市。有多个城市，每个城市可以有多个位置。此外，卡车或飞机可以携带的包裹数量没有限制（理论上，卡车或飞机可以携带无限数量的包裹）。

### A. 8. 1 Load a package into a truck

| Action description                                           |
| ------------------------------------------------------------ |
| This action enables the agent to load a package into a truck. For example, load a package_1 into a truck_1. |
| **GPT-4: Load a package into a truck**                       |
| Parameters:<br/>1. ?p - package: the package to load<br/>2. ?t - truck: the truck to load the package into<br/>3. ?l - location: the location where the package and truck are<br/>Preconditions:<br/>(and<br/>(package-at ?p ?l)<br/>(truck-at ?t ?l)<br/>)<br/>Effects:<br/>(and<br/>(not (package-at ?p ?l))<br/>(package-in-truck ?p ?t)<br/>) |

---

> Comments: 后面还有很多提示词（。。。附录总共是58页，大半部分是在展示自己的提示词。阅读这些提示词主要是了解这些提示词的书写和结构吧，因为自己设计系统的话对象肯定不一样惹。

