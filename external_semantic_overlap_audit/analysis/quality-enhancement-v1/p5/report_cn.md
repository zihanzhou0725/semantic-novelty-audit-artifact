# P5 Certified Exposure Interval 验证报告

## 结论

- **P5 Theory-Contribution Go**：显式假设、完整证明、可证伪 bound、两个公共 benchmark 实例和实验设计后果全部满足。
- 该命题识别 protocol exposure 区间，不识别 behavioral exploitation 或因果效应。

## 命题

对两两不相交的 certificate groups `(G_j, lower_j, upper_j)`，总 exposure 满足：

`sum(lower_j) <= C(E) <= uncovered + sum(upper_j)`。

该区间相对于已验证 certificate system 是 sharp 的，并通过 assignment checker 与重叠组拒绝规则进行可执行证伪。

## 实例

- Tensor Logic endpoint-conditioned semantics：`[484, 1000] / 1000`。
- Tensor Logic relation schema：`[999, 1000] / 1000`。
- Occam Fig. 2：`[2, 4] / 20`。
- Occam Fig. 3：`[2, 4] / 20`。
- Occam 全训练 seed-support 共复算 75,093 个 signatures，未发现跨 seed transition-tensor collision。

## 设计后果

- prevalence 必须报告区间，`no witness` 不得计作 novel。
- split mutation 后必须重新验证 executable witnesses。
- online generator 必须持久化 semantic IDs 与 category assignments。
- protocol exposure 与 behavioral effect 必须使用不同 estimand。

P5 Go 不改变 P3 Breadth No-Go，也不允许 memorization 因果表述。
