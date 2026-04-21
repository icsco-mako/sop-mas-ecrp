# 审稿建议

**邮件标题：** Rejection Decision on paper ID 252923338  
**发件人：** "International Journal of Production Research" <onbehalfof@manuscriptcentral.com>  
**发送日期：** 2025-11-09 19:00:02  
**收件人：** jinzhihong@dlmu.edu.cn  

---

**正文内容：**

09-Nov-2025  
Dear Prof. Zhihong Jin,  

Ref: # TPRS-2025-IJPR-3131 entitled "Optimization of Empty Container Repositioning by Integrating Large Language Models and Multi-Agent Systems" by Peng, Kangzhen; Xu, Shida; Wang, Yuhong; Jin, Zhihong; Wang, Zhenkai; Hua, Mengying that you submitted to the International Journal of Production Research.  

I am sorry to have to advise you that our referees have now considered your paper and unfortunately feel it unsuitable for publication in International Journal of Production Research. For your information I attach the reviewers' comments at the bottom of this email. Please note that fuller review reports may also be attached to this email.  

Please note that for IJPR papers, the following elements are mandatory:  
- an exhaustive analysis of related production research publications (especially in IJPR);  
- a novel decision aid model for design or management of production systems or/and their logistics, the model should be explained for a large audience in production research;  
- comparisons with the state of the art;  
- discussion on real life applications of the proposed approach in production systems and logistics;  
- managerial insights for decision makers in industry;  
- research perspectives.  

See all recent related articles published in IJPR before any revision and new submission:  
https://www.tandfonline.com/journals/tprs20  
and the attached file with advice to authors.  

I hope you will find them to be constructive and helpful. You are of course now free to submit the paper elsewhere should you choose to do so.  

Thank you for considering the International Journal of Production Research for the publication of your research. I hope the outcome of this specific submission will not discourage you from the submission of future manuscripts.  

Sincerely,  
Alexandre Dolgui  
Editor-in-Chief, International Journal of Production Research  
alexandre.dolgui@imt-atlantique.fr  

---

**Editor Comments to Author:**  

**Associate Editor Comments to the Author:**  
The current version is under the bar for IJPR.  

---

**Reviewer(s)' Comments to Author:**  

**Reviewer: 1 Comments to the Author**  
This manuscript shows an excellent contribution to the field of operations research and supply chain optimization. Especially, bridging the application of LLM and MAS can be a motivated insights for solving supply chain problems. Please refer some minor suggestions below for this manuscript:  
1. Check minor grammatical inconsistencies (e.g., "dependent on experts who manually translating" → "depend on experts who manually translate")  
2. Check some typos (e.g., "Problem Descriptin"→"Problem Description" in Figure 1)  
3. Check the use of abbreviation. You only need to spell out the full term the first time you use it, followed by the abbreviation in parentheses, and then use the abbreviation consistently afterward)  
4. Strengthen the transition between the LLM–MAS framework and the ECR application. (explicitly highlight why ECR was chosen as the demonstration case)  
5. Add a brief table to summarize key previous studies (method, domain, limitation) to make the contribution stand out more clearly.  

**Reviewer: 2 Comments to the Author**  
The comments I gave are as below:  

Comments on IJPR paper “Optimization of Empty Container Repositioning by Integrating Large Language Models and Multi-Agent Systems”  

**Good part**  
This paper proposes a practical agent-based process for ECR. Notably, it includes a business analysis agent (BA-Agent) that transforms solver logs and model outputs into decision-oriented narratives and dashboards. This design addresses the fact that business stakeholders often struggle to understand the rationale behind optimization results. This shortens the decision cycle through clearer, auditable explanations.  

In the evaluation, the authors benchmarked several popular prompting/agent strategies: Standard Prompt (SPM), Chain of Thought (CoT), Chain of Experts (CoE), and OptiMUS, under a consistent task setting. This comparative perspective helps clarify the benefits of orchestration choices compared to baseline prompting.  

**Challenging part**  
The paper's innovation and contributions fall short of reaching IJPR status. The proposed SOP-MAS framework primarily integrates well-known components, such as structured prompts, role specialization, self-consistency/cross-verification, and tool-in-the-loop retesting. It does not introduce any significantly novel algorithmic ideas, nor does it demonstrate significant additional benefits over baselines.  

There are still some improvements need to be made for this work especially in the ‘Methodology’ and ‘System Implementation and Experimental Analysis’ sections, the reproducibility of the method is vague and some conclusions are too arbitrary which I will present in the following.  

In the section 3.2, while the paper states that the SOP establishes quality standards for intermediate outputs, it seems not provide explicit, testable gate criteria that how to decide whether an intermediate output is acceptable and the workflow may proceed to the next agent. The flow terminates when the current outcome is deemed “correct,” but “correct” is triggered by an agent-side signal paired with a free-text error description. There is no externally verifiable test in the steps that defines correctness. That makes termination contingent on self-report rather than measurable gates, which is brittle and hard to audit. Futhermore, the mechanism rolls back to the last agent and evaluates whether its output was the source of the exception, then prunes the message pool and action history. But the paper does not specify diagnostic criteria for blame assignment. Without formal checks, backtracking may oscillate, remove useful context, or misattribute faults.  

In section 3.3, to mitegate hallucination, the ingredients (structured prompting, role specialization, self-consistency/cross-checks, tool-in-the-loop retries) are widely known; the paper does not isolate a new algorithmic idea or provide ablations showing a unique additive effect beyond these baselines.  

In section 4.1 performance measurement metrics, the authors used CER, RER ,and AR to measure whether the code could compile, run, or pass the test case. They did not directly verify whether the generated mathematical model accurately satisfied all tasks or mathematical constraints. However, in Section 3, Step 5: Iterative Modeling, they say, " If the outcome is correct, advance to Step 7." I am confused about how they measure the correctness of the modeling results. Although the Section 3 mentions using JSON structured output, SOP role constraints, and exception backtracking to reduce illusions and perform cross-verification, these are mechanistic descriptions and are not translated into machine-checkable indicators and pass rates, such as the feasible-solve rate, MIP gap, constraint-violation rate, accuracy pass rate of variable / constraint target setting and the violation rate of prohibiting new parameters.  

In section 4.2, This paper does not provide a transparent runtime and quantify the latency or opportunity costs for each agent and the end-to-end process. This prevents readers from assessing deployability, scalability, or the accuracy-latency trade-offs at different collaboration thresholds 𝐾. It seems unsure whether the approach is practical beyond demonstrations.  

In section 4.3, the evaluation fixes the LLM model to DeepSeek-v3-0324 and agentic strategies; please include cross-model robustness (e.g., GPT-4 class, Claude class, and Gemini) to decouple gains from model choice. Additionally, please replicate the main results on at least two MILP solvers (e.g., Gurobi, CPLEX) and maintain consistent time limits.  

**In mathematical formulation:**  
• It could be clear if the article use mathematical formulation rather than latex version.  
• What’s the definition of \delta_{i \in H} + \delta_{i,j \in L}?  
• In page 32 line 57-60, addition is used throughout the big-m constraints, does the article mean that even if repos_mode is disabled, it can still be allowed by the lambda item?  
• F_max^k is more like the maximum capacity of the entire network on a regular basis (rather than each arc having its own share of maximum F_{ij,k}^{max}).  

**About the paper writing, there are some errors that the authors need to check carefully, which include but not limited to:**  
• In page 10 line 23, “Drawing inspiration from the work of (Hong et al., 2024)”, please use correct author–year citation.  
• In page 11 line 19, there is a grammar error.  
• There are many extra spaces in this article, like page 11 line 5, page 16 line 6…  
• Several figures appear low-resolution with small and difficult to read.  

**Reviewer: 3 Comments to the Author**  
The paper in the current version is not acceptable. The OR Model is unclear, no explanation about the details of datasets used in the study.  
Unsure that this paper matches well the IJPR requirements.  

_______________________________________________________

# 审稿建议翻译

**邮件标题：** 关于稿件编号 252923338 的拒稿决定
**发件人：** “International Journal of Production Research” [onbehalfof@manuscriptcentral.com](mailto:onbehalfof@manuscriptcentral.com)
**发送日期：** 2025年11月9日 19:00:02
**收件人：** [jinzhihong@dlmu.edu.cn](mailto:jinzhihong@dlmu.edu.cn)  

------

**正文内容：**

2025年11月9日  

尊敬的金志宏教授：  

您提交至《International Journal of Production Research》（国际生产研究期刊）的稿件，编号为 #TPRS-2025-IJPR-3131，题为《通过整合大语言模型与多智能体系统优化空箱调运》（Optimization of Empty Container Repositioning by Integrating Large Language Models and Multi-Agent Systems），作者为彭康振、徐诗达、王宇虹、金志宏、王振凯、华梦莹。  

很遗憾地通知您，我们的审稿人已对您的稿件进行了评审，一致认为该稿件目前尚不适合在《International Journal of Production Research》发表。为便于您了解评审意见，我们在本邮件末尾附上了审稿人的详细评论。请注意，更完整的审稿报告可能也作为附件随本邮件一并发送。  

请注意，IJPR 对投稿论文有以下强制性要求：  

- 必须对相关生产研究领域的文献（尤其是发表在 IJPR 上的文献）进行详尽分析；  
- 必须提出一种新颖的决策支持模型，用于生产系统及其物流的设计或管理，并且该模型应能被广大生产研究领域的读者理解；  
- 必须与现有最先进方法进行比较；  
- 必须讨论所提方法在实际生产系统与物流中的应用；  
- 必须为工业界决策者提供管理洞见；  
- 必须提出未来的研究方向。

在修改稿件或重新投稿前，请务必查阅 IJPR 近期发表的相关文章：
https://www.tandfonline.com/journals/tprs20
以及随附的作者投稿指南文件。  

我们希望这些意见对您具有建设性和帮助性。当然，您现在可以自由选择将该稿件投往其他期刊。  

感谢您考虑将《International Journal of Production Research》作为您研究成果的发表平台。希望此次投稿结果不会影响您未来向本刊提交新稿件的积极性。  

此致
敬礼！  

Alexandre Dolgui
《International Journal of Production Research》主编
[alexandre.dolgui@imt-atlantique.fr](mailto:alexandre.dolgui@imt-atlantique.fr)  

------

**编辑给作者的意见：**

**副主编意见：**
当前版本未达到 IJPR 的发表标准。

------

**审稿人意见：**

**审稿人 1 意见：**
本稿件在运筹学与供应链优化领域展现了出色的贡献。特别是将大语言模型（LLM）与多智能体系统（MAS）相结合，为解决供应链问题提供了富有启发性的思路。以下是一些小建议供参考：  

1. 请检查一些语法不一致之处（例如：“dependent on experts who manually translating” 应改为 “depend on experts who manually translate”）；  
2. 请修正拼写错误（例如：图1中的 “Problem Descriptin” 应为 “Problem Description”）；  
3. 请规范缩写使用方式：首次出现时应写出全称并附上括号内的缩写，之后统一使用缩写；  
4. 请加强 LLM–MAS 框架与空箱调运（ECR）应用场景之间的衔接，明确说明为何选择 ECR 作为案例演示；  
5. 建议增加一个简表，总结已有关键研究（方法、领域、局限性），以更清晰地凸显本文的贡献。

**审稿人 2 意见：**
我对该稿件的评述如下：

**优点部分：**
本文提出了一个面向空箱调运（ECR）的实用型基于智能体的流程。特别值得一提的是，其中包含一个业务分析智能体（BA-Agent），可将求解器日志和模型输出转化为面向决策的叙述与可视化仪表盘。这一设计解决了业务利益相关者常常难以理解优化结果背后逻辑的问题，从而通过更清晰、可审计的解释缩短了决策周期。  

在实验评估中，作者在统一任务设定下，对多种主流提示/智能体策略进行了基准测试，包括标准提示（SPM）、思维链（CoT）、专家链（CoE）和 OptiMUS。这种对比视角有助于厘清不同协作编排策略相较于基线提示的优势。

**存在问题部分：**
本文的创新性与贡献尚未达到 IJPR 的发表门槛。所提出的 SOP-MAS 框架主要整合了若干已知组件，如结构化提示、角色专业化、自洽性/交叉验证、工具闭环重试等，并未引入显著新颖的算法思想，也未展现出相对于基线方法的实质性性能提升。

尤其在“方法论”和“系统实现与实验分析”部分，仍需进一步改进。当前方法的可复现性较模糊，部分结论显得武断，具体如下：

- 在第 3.2 节中，尽管文中提到 SOP 为中间输出设定了质量标准，但并未提供明确、可检验的“关卡”标准来判断中间输出是否可接受、流程是否可进入下一智能体。流程终止条件是当前结果被判定为“正确”，但“正确”仅由智能体端发出信号并附带自由文本错误描述触发。整个过程中缺乏外部可验证的正确性判定机制，导致流程终止依赖于自我报告而非可度量的关卡，这使得系统脆弱且难以审计。此外，当发生异常时，机制会回滚至上一智能体，并评估其输出是否为异常源头，随后剪枝消息池与动作历史。但文中未说明如何诊断归因。若无形式化检查机制，回溯过程可能出现震荡、误删有用上下文或错误归责等问题。
- 在第 3.3 节中，为缓解幻觉问题所采用的技术（结构化提示、角色专业化、自洽/交叉检查、工具闭环重试）均为广为人知的方法；文章既未提炼出新的算法思想，也未通过消融实验证明这些组合带来了超越基线的独特增益。
- 在第 4.1 节性能度量指标中，作者使用 CER、RER 和 AR 来衡量生成代码能否编译、运行或通过测试用例，但未直接验证所生成数学模型是否准确满足所有任务要求或数学约束。然而在第 3 节步骤 5（迭代建模）中却写道：“若结果正确，则进入步骤 7。” 我对此处“正确性”的衡量方式感到困惑。尽管第 3 节提到了使用 JSON 结构化输出、SOP 角色约束和异常回溯机制来减少幻觉并进行交叉验证，但这些仅为机制性描述，未转化为机器可检验的指标与通过率，例如：可行解率、MIP 间隙、约束违反率、变量/约束目标设定的准确通过率，以及禁止引入新参数的违反率等。
- 在第 4.2 节中，本文未提供透明的运行时间数据，也未量化各智能体及端到端流程的延迟或机会成本。这使得读者无法评估该方法的可部署性、可扩展性，或在不同协作阈值 𝐾 下的精度-延迟权衡。因此，该方法是否具备超出演示场景的实际可行性尚不明确。
- 在第 4.3 节中，实验固定使用 DeepSeek-v3-0324 作为大语言模型及特定智能体策略；建议补充跨模型鲁棒性实验（例如 GPT-4 系列、Claude 系列、Gemini），以剥离模型选择带来的性能增益。此外，请至少在两种 MILP 求解器（如 Gurobi、CPLEX）上复现主要结果，并保持一致的时间限制。

**关于数学建模：**
• 若文章直接使用数学公式而非 LaTeX 代码形式呈现，会更清晰；
• \delta_{i \in H} + \delta_{i,j \in L} 的定义是什么？
• 第 32 页第 57–60 行，在 Big-M 约束中全程使用加法，是否意味着即使 repos_mode 被禁用，仍可通过 lambda 项允许调运？
• F_max^k 更像是整个网络在常规情况下的最大容量（而非每条弧拥有各自的 F_{ij,k}^{max} 上限）。

**关于写作方面，作者需仔细检查以下错误（但不限于）：**
• 第 10 页第 23 行：“Drawing inspiration from the work of (Hong et al., 2024)”，请使用正确的作者–年份引用格式；
• 第 11 页第 19 行存在语法错误；
• 文中多处存在多余空格，例如第 11 页第 5 行、第 16 页第 6 行等；
• 多幅图表分辨率较低，字体过小，难以辨认。

**审稿人 3 意见：**
当前版本的稿件不可接受。运筹学模型表述不清，且未说明研究所用数据集的详细信息。
不确定该稿件是否符合 IJPR 的投稿要求。

------

**以下问题有助于您准备符合 IJPR 要求的稿件：**

Q1：本文是否对生产研究领域的文献作出了新颖且重要的贡献？  

Q2：本文是否提供了所提方法在生产系统中实际或潜在应用的证据？  

Q3：是否充分致谢了该领域的其他贡献者？参考文献是否足够完整？  

Q4：本文是否恰当地将所提方法与已发表文献中的方法进行了性能比较？  

Q5：本文是否说明了作者未来的研究计划？  

Q6：从标题和摘要是否能清晰了解论文的性质与内容？  

Q7：论文是否表述清晰、简洁、准确且逻辑严谨？  

Q8：论文是否可通过精简或扩展加以改进？  

Q9：论文主题是否与生产研究相关？是否符合 IJPR 近期发表趋势及期刊政策？  

Q10：所有参考文献是否都相关？如有不相关者，请删除。  



