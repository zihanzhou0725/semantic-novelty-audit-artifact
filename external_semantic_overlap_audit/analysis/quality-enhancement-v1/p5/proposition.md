# Certified Exposure Interval Proposition

版本：`quality-enhancement-p5-v1`

## 1. 目标量

固定一个语义命名空间和一个 evaluation 集合

\[
E=\{e_1,\ldots,e_N\}.
\]

对每个 evaluation identity 定义真实 exposure 指示量

\[
z_i=\mathbf{1}\{e_i\text{ 的同语义 identity 在训练中出现}\}\in\{0,1\},
\]

总 exposure 数为

\[
C(E)=\sum_{i=1}^{N}z_i.
\]

不同语义命名空间必须分开分析。例如，Tensor Logic 的 direct triple、
endpoint-conditioned two-hop path 和 relation-composition schema 是三个不同目标，
不能共用一个 \(z_i\)。

## 2. 显式假设

**A1（有限、冻结分母）**：\(E\) 在查看 audit 结果前冻结，且 \(N<\infty\)。

**A2（确定语义身份）**：每个 \(e_i\) 在指定命名空间中有确定的 canonical
semantic ID。该 ID 可以由完整规范化或 sound executable witness 支持，但 surface
string、seed 或参数名本身不能替代语义身份。

**A3（sound certificate）**：第 \(j\) 个可执行 certificate 给出一个非空索引组
\(G_j\subseteq\{1,\ldots,N\}\) 和整数
\(0\leq \ell_j\leq u_j\leq |G_j|\)，并保证真实 assignment 满足

\[
\ell_j\leq\sum_{i\in G_j}z_i\leq u_j.
\]

每个约束必须有确定性 verifier；未找到 witness 不能自动产生 \(z_i=0\) 约束。

**A4（组不相交）**：\(G_1,\ldots,G_J\) 两两不相交。若原始证书重叠，必须先
atomize 为不相交组，或改用一般整数规划；不能直接相加。

**A5（未覆盖项不加假设）**：令

\[
R=\{1,\ldots,N\}\setminus\bigcup_{j=1}^{J}G_j.
\]

除 \(z_i\in\{0,1\}\) 外，不对 \(i\in R\) 添加约束。

## 3. 命题：证书暴露区间

在 A1-A5 下，定义

\[
L=\sum_{j=1}^{J}\ell_j,\qquad
U=|R|+\sum_{j=1}^{J}u_j.
\]

则

\[
\boxed{L\leq C(E)\leq U}
\]

且 prevalence 满足

\[
\boxed{\frac{L}{N}\leq \frac{C(E)}{N}\leq\frac{U}{N}}.
\]

该区间相对于 certificate system 是 sharp 的：只使用 A1-A5 和已验证约束时，
不存在严格更大的通用下界或严格更小的通用上界。

## 4. 完整证明

由 A4，所有 \(G_j\) 两两不相交，且与 \(R\) 不相交，因此

\[
C(E)=
\sum_{j=1}^{J}\sum_{i\in G_j}z_i+\sum_{i\in R}z_i.
\]

由 A3，

\[
\sum_{i\in G_j}z_i\geq\ell_j
\]

对每个 \(j\) 成立；由 A5，\(\sum_{i\in R}z_i\geq0\)。相加得到

\[
C(E)\geq\sum_{j=1}^{J}\ell_j=L.
\]

同理，由 A3，

\[
\sum_{i\in G_j}z_i\leq u_j,
\]

而二元性给出

\[
\sum_{i\in R}z_i\leq |R|.
\]

相加得到

\[
C(E)\leq\sum_{j=1}^{J}u_j+|R|=U.
\]

因为 \(N>0\)，两端除以 \(N\) 得 prevalence 区间。

下面证明相对于证书信息的 sharpness。对每个 \(G_j\)，任选恰好 \(\ell_j\)
个索引令 \(z_i=1\)，其余令 \(z_i=0\)；对所有 \(i\in R\) 令 \(z_i=0\)。
由于 \(0\leq\ell_j\leq|G_j|\)，该 assignment 存在并满足所有 certificate，
且总数恰为 \(L\)。因此仅凭这些 certificate 不能推出大于 \(L\) 的通用下界。

类似地，对每个 \(G_j\) 任选恰好 \(u_j\) 个索引令 \(z_i=1\)，并对所有
\(i\in R\) 令 \(z_i=1\)。该 assignment 满足所有 certificate，且总数恰为
\(U\)。因此仅凭这些 certificate 不能推出小于 \(U\) 的通用上界。证毕。

这里的 sharpness 是逻辑上的 partial-identification sharpness，不声称两个极端
assignment 都能由原始训练随机过程以正概率生成。加入新的、经验证的结构约束后，
区间可以进一步收紧。

## 5. 推论：未记录类别分配下的 grouped-seed bound

设一个固定 seed \(s\) 对应 \(K\) 个 evaluation identities：

\[
G_s=\{\phi(1,s),\ldots,\phi(K,s)\},
\]

其中 \(\phi(c,s)\) 是 category \(c\) 与 seed \(s\) 生成的 canonical task。
假设：

1. 每个使用 seed \(s\) 的训练 episode 恰好选择 \(G_s\) 中一个 category；
2. 训练中 seed \(s\) 出现 \(m_s\) 次；
3. category assignment 未记录；
4. exact semantic-signature census 已排除其他 seeds 对 \(G_s\) 的额外碰撞。

若 \(m_s=0\)，则 \(|G_s\cap T|=0\)。若 \(m_s>0\)，至少一个 category 被选中，
至多 \(\min(K,m_s)\) 个不同 categories 被选中，所以

\[
\boxed{
1\leq\sum_{i\in G_s}z_i\leq\min(K,m_s)
}.
\]

这是 A3 的一个可执行 certificate。若未来持久化了 category assignments，
上下界可直接收紧为出现过的不同 categories 数量。

## 6. 可证伪审计规则

命题通过以下条件被直接证伪或拒绝应用：

1. denominator 在结果检查后改变；
2. semantic ID canonicalization 不确定或混合了不同 claim scopes；
3. 某个 certificate verifier 失败；
4. certificate groups 重叠但仍直接相加；
5. 枚举出的真实 assignment 违反任一 \([\ell_j,u_j]\)；
6. grouped-seed 应用中发现跨 seed semantic-signature collision，却未扩展约束组。

实现 `solve_disjoint_certificate_bounds` 会拒绝越界或重叠组；
`assignment_satisfies_constraints` 提供第 5 条的 executable falsification test。

## 7. 两个公共 benchmark 实例

### 7.1 Tensor Logic FB15k-237

对 endpoint-conditioned two-hop semantics，\(N=1000\)。484 个 test paths
存在 retained executable witness，因此形成 484 个 singleton constraints
\([1,1]\)；其余 516 项保持未覆盖。命题给出

\[
C_{\text{endpoint}}\in[484,1000].
\]

对 relation-composition schema，999 个 test paths 有 training schema witness，
得到

\[
C_{\text{schema}}\in[999,1000].
\]

direct triple identity 的完整 membership census 则给出 \(C_{\text{direct}}=0\)。
这三个结果必须分开报告：direct-edge disjointness 不推出 endpoint 或 schema
semantic disjointness。

### 7.2 In-Context Occam's Razor Markov experiments

每个主实验有 \(N=20\) 个 eval latent tasks，即 10 个 seeds 乘 2 个 orders。
完整 transition-tensor signature census 只发现 seeds `5` 和 `12579` 位于训练
支持；未发现跨 seed tensor collision。

对每个支持 seed，两个 order-specific tasks 构成 \(K=2\) 的组。训练 occurrence
分别为 \(m_5=6\) 和 \(m_{12579}=32\)，由 grouped-seed 推论，每组约束均为
\([1,2]\)。其余 16 项由 signature census 给出 singleton \([0,0]\)。因此
Fig. 2 和 Fig. 3 分别得到

\[
C_{\text{Markov}}\in[2,4],\qquad
\frac{C_{\text{Markov}}}{20}\in[10\%,20\%].
\]

该结论识别 protocol exposure，不识别 exposure 对模型输出的因果效应。

## 8. 对实验设计与解释的改变

1. 所有 prevalence 结果必须报告 \([L,U]\)，禁止将 `no witness` 计为 novel。
2. 图数据经删除或 scrub 后必须重新运行 executable witness verifier；不能沿用
   mutation 前的路径声明。
3. online task generator 必须持久化 task semantic ID 与 category assignment，
   并在训练前冻结与 eval semantic IDs 的 disjointness 检查。
4. exposure certificate 与 behavioral estimand 分开：前者只能改变 protocol
   contamination 解释，不能单独支持 memorization 或 accuracy-effect claim。
