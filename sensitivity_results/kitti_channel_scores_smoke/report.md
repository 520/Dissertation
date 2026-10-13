# KITTI YOLOv8n 通道重要性评分

状态：SUBSET_ONLY；评分图像：2/5985；卷积层：64；输出通道：5417。

## 评分定义与来源

- 通道从 0 开始编号。排名从 1 开始，1 表示最高分；相同分数共享排名。
- L1 = Σ |W[c,i,h,w]|；L2 = √Σ W[c,i,h,w]²。使用与 ONNX 对应的 BN 融合后卷积权重，不包括偏置。
- Taylor-1 = 全训练图像平均 |Σ A[c,h,w] × ∂检测损失/∂A[c,h,w]|。A 是融合卷积输出、激活函数之前；每张图像独立反向传播，先取绝对值再平均。
- 对融合卷积，该 Taylor-1 等价于 |Σ W×∂L/∂W + b×∂L/∂b|；另提供 taylor1_weight_only，排除偏置。
- 检测损失 = 7.5×box + 0.5×classification + 1.5×DFL。仅使用训练集真实标签，640×640，batch=1，FP32，无数据增强，无优化器更新。
- engine 没有训练反向图：Taylor 在权重已与 ONNX 核对的可求梯度 PyTorch 图上计算。引擎与 PyTorch 在三张真实图像上做输出对照，误差和严格比较结果见 manifest；内部权重无法恢复，因此不能宣称这就是 engine 的精确梯度。
- 各 TensorRT 融合层与源卷积可能多对一，映射见 engine_layers；Concat、SiLU、Reformat 等无独立卷积核权重，不定义权重 L1/L2。
- DFL 固定投影不参与该训练损失的解码路径：Taylor 标为不可用，不能解释为不重要。最终检测输出通道以及 DFL 通道不应直接剪除。
- 跨层原始 L1/L2 受滤波器尺寸影响：同时给出 l1_mean_abs、l2_rms；层内排名更适合比较同一层通道。
- 层级 Taylor 是各通道绝对一阶分数之和，不是同时删除整层的损失变化。分数不是实测 mAP 下降。

## Taylor-1 层排序（前 20，降序）

| 排名 | 源卷积 | 通道数 | 通道分数之和 | 通道平均分数 |
|---:|---|---:|---:|---:|
| 1 | model.22.cv2.1.2 | 64 | 12.669565 | 0.19796195 |
| 2 | model.9.cv1.conv | 128 | 6.2939674 | 0.04917162 |
| 3 | model.3.conv | 64 | 4.3713335 | 0.068302087 |
| 4 | model.22.cv2.0.2 | 64 | 4.3113794 | 0.067365303 |
| 5 | model.4.cv1.conv | 64 | 4.1822493 | 0.065347645 |
| 6 | model.2.cv2.conv | 32 | 3.9957189 | 0.12486621 |
| 7 | model.6.cv1.conv | 128 | 3.5346906 | 0.02761477 |
| 8 | model.2.cv1.conv | 32 | 3.5253137 | 0.11016605 |
| 9 | model.4.cv2.conv | 64 | 3.4954574 | 0.054616523 |
| 10 | model.6.cv2.conv | 128 | 3.3639253 | 0.026280666 |
| 11 | model.15.cv1.conv | 64 | 3.2709581 | 0.051108721 |
| 12 | model.5.conv | 128 | 3.2512218 | 0.025400171 |
| 13 | model.1.conv | 32 | 2.9048533 | 0.090776665 |
| 14 | model.12.cv1.conv | 128 | 2.8855562 | 0.022543408 |
| 15 | model.8.cv1.conv | 256 | 2.7779897 | 0.010851522 |
| 16 | model.18.cv1.conv | 128 | 2.7068659 | 0.02114739 |
| 17 | model.8.cv2.conv | 256 | 2.7053425 | 0.010567744 |
| 18 | model.7.conv | 256 | 2.5928846 | 0.010128455 |
| 19 | model.0.conv | 16 | 2.3944495 | 0.14965309 |
| 20 | model.12.cv2.conv | 128 | 2.3362909 | 0.018252272 |

## 数据与文件

- channels.csv：每层每通道的全部分数、层内排名及全局排名。
- channels_ranked_l1/l2/taylor1.csv：三个指标各自的全局降序排序。
- layers_ranked_l1/l2/taylor1.csv：三个指标各自的层级降序排序。
- per_layer/*.csv：按源层名分文件，可按 rank_within_layer_* 排序。
- manifest.json / images.csv：数据覆盖、样本哈希、版本、公式和验证信息。

## 方法参考

- https://arxiv.org/abs/1611.06440
- https://docs.ultralytics.com/reference/utils/loss/
- https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip
