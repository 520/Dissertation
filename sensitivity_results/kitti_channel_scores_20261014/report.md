# KITTI YOLOv8n 通道重要性评分

状态：COMPLETE；评分图像：5985/5985；卷积层：64；输出通道：5417。

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
| 1 | model.22.cv2.0.2 | 64 | 8.6562048 | 0.1352532 |
| 2 | model.22.cv2.1.2 | 64 | 7.4257566 | 0.11602745 |
| 3 | model.9.cv1.conv | 128 | 4.3283192 | 0.033814993 |
| 4 | model.3.conv | 64 | 3.7059303 | 0.05790516 |
| 5 | model.15.cv1.conv | 64 | 3.6706378 | 0.057353715 |
| 6 | model.4.cv1.conv | 64 | 3.293706 | 0.051464156 |
| 7 | model.2.cv2.conv | 32 | 3.0194117 | 0.094356617 |
| 8 | model.4.cv2.conv | 64 | 2.9964745 | 0.046819914 |
| 9 | model.2.cv1.conv | 32 | 2.575753 | 0.080492282 |
| 10 | model.12.cv1.conv | 128 | 2.5371087 | 0.019821161 |
| 11 | model.15.cv2.conv | 64 | 2.5057896 | 0.039152962 |
| 12 | model.6.cv1.conv | 128 | 2.4326679 | 0.019005218 |
| 13 | model.6.cv2.conv | 128 | 2.3815173 | 0.018605604 |
| 14 | model.5.conv | 128 | 2.3249758 | 0.018163873 |
| 15 | model.1.conv | 32 | 2.1950768 | 0.068596151 |
| 16 | model.12.cv2.conv | 128 | 2.1643293 | 0.016908823 |
| 17 | model.8.cv1.conv | 256 | 1.868205 | 0.0072976756 |
| 18 | model.8.cv2.conv | 256 | 1.8093923 | 0.0070679387 |
| 19 | model.9.cv2.conv | 256 | 1.7205655 | 0.006720959 |
| 20 | model.0.conv | 16 | 1.6998281 | 0.10623925 |

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
