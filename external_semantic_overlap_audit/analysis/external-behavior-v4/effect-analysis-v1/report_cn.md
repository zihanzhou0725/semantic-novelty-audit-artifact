# Directional V4 暴露效应解盲报告

## 主结果

- 暴露减未暴露准确率效应：23.3391 pp。
- 95% 层级 bootstrap 区间：[17.7498, 29.2257] pp。
- 预注册判定：`directional_positive_go`。

## 描述性结果

- Cross-entropy 暴露减未暴露：-1.493775。
- Brier 暴露减未暴露：-0.422398。
- ECE-15 暴露减未暴露：-0.201770。

Gate 仅使用冻结的 accuracy estimand 与 directional rule。block、seed、pattern、relation 和 calibration 结果均为描述性。
