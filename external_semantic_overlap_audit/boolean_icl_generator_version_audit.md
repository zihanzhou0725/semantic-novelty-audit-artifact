# Boolean ICL 生成器版本核验

核验日期：2026-10-07

## 最终判定

公开可复现的正式生成器已经确认：

- 仓库：`https://github.com/satwik77/incontext-bool`
- HEAD：`a25a2cb0c62ad541cd30b0672df136230d878d9c`
- 首次加入代码的 commit：
  `8726c19bf6bc813f48373a1497372a35404b42da`
- 首次代码提交时间：2024-04-02

camera-ready PDF 明确将该仓库声明为 source code。ICLR proceedings 的最终
Supplementary ZIP、OpenReview camera-ready revision 附件和此前下载的
OpenReview 当前附件逐字节相同：

```text
SHA-256
fb3e9f4eca33acba5504cf23cbaaac1a1b8c735edf5871e17731d94604dacfe8
```

仓库的 `src/bool_tasks.py` 从首次代码 commit 到 HEAD 没有修改。与
2023-09-29 Supplementary ZIP 比较时，核心生成器中：

- `Disjunction`
- `SparseDisjunction`
- `SparseThreshold`
- `IntHalfspace`
- `Majority`
- `Parity`
- `SparseParity`
- `DNF`
- `TeachDNF`
- `CNF`

的 AST 完全相同。`Conjunction.__init__` 只删除了一个不参与当前执行路径的
`self.p` 死变量；实际权重采样、输入采样和 `evaluate` 逻辑未改变。
`train()`、`eval_batch_bool()` 和 `eval_model_bool()` 的 AST 也完全相同。

因此，之前发现的生成器行为不是匿名旧附件后来已经修复的问题。

## 公开代码与 camera-ready 论文的冲突

| 任务 | camera-ready 论文 | 正式公开代码 |
|---|---|---|
| Sparse Disjunction | `n=50`, `k<=3`，signed literals，uniform functions | 硬编码 `k=4`，仅 positive literals |
| Sparse Parity | 主表含 `(10,2)` 与 `(20,3)` | `SparseParity` 硬编码 `k=2`，不接收 `k` |
| 0-1 Threshold | 权重按各自范围均匀，阈值为 `{-3,...,3}` | 权重概率 `[0.7,0.15,0.15]`，六个半整数阈值 |
| task registry | 主表使用 0-1 Threshold / Integer Halfspace | registry 名称为 `sparse_thres` / `int_halfspace` |

正式仓库只有 `main` 分支，没有 tag；全部历史只有 4 个 commits。代码加入后，
后续 commit 只修改 README 图片。没有公开分支能够生成论文中的
Parity-(20,3)，也没有公开版本把 Sparse Disjunction 改成论文定义的 `k=3`。

## 能确认与不能确认的内容

### 能确认

1. 作者指定的正式公开实现就是 `satwik77/incontext-bool@a25a2cb`。
2. 它与 camera-ready Supplementary 中的生成器功能等价。
3. 公开实现使用 same-distribution fresh sampling，没有 function-disjoint
   train/test protocol。
4. 代码口径下的 truth-table coverage 计算可以作为可复现结论。

### 不能确认

1. 主表冲突任务究竟使用了未发布分支、临时手工修改，还是表格标签/论文描述
   有误。
2. 每个主表模型实际停止在哪个训练 step。论文只给出 `30k--500k` 范围、
   Conjunction 的 `20k x 64` 示例和“最多 40k”，正式仓库 README 则只提供
   `15k x 64` 的演示命令。
3. Parity 的 `500k` steps 对应的实际 batch size 和最终 function draws。
4. 没有公开 checkpoint、训练日志、逐任务配置或逐题 evaluation manifest
   可以恢复上述信息。

所以必须区分：

- **released-code generator**：已确认，可精确复现；
- **paper-defined generator**：可按文字重新实现；
- **main-table actual generator**：公开证据不足，不能唯一确定。

论文中不能再把统一 `N=1.28M` 的覆盖率称为所有任务的“实际 overlap”。
它只能作为固定预算下的可比情景；只有 Conjunction 的 `20k x 64` 是
camera-ready 论文明确给出的实例。

## 来源

- ICLR proceedings paper:
  `f7e7fabd73b3df96c54a320862afcb78-Paper-Conference.pdf`
  - SHA-256:
    `5af0615bf431cc637d93dae8f1f60337dbfd44b4b610a8c2832a18716c9b617f`
- ICLR proceedings supplementary:
  `f7e7fabd73b3df96c54a320862afcb78-Supplementary-Conference.zip`
  - SHA-256:
    `fb3e9f4eca33acba5504cf23cbaaac1a1b8c735edf5871e17731d94604dacfe8`
- OpenReview camera-ready edit ID: `GWG4gLNePy`
- GitHub repository HEAD:
  `a25a2cb0c62ad541cd30b0672df136230d878d9c`
- Public `bool_tasks.py` SHA-256:
  `5bda5cdb00bce9ad7710faa643d822c99eb09ba20a5de1a26d9da3788269ca60`
