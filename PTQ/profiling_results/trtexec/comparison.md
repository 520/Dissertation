# trtexec Top 10 cross-check

TensorRT 11.3.0.99 trtexec, original plan, real fixed input, batch 1, CUDA Graph

Three alternating runs per model, same fixed input as Python IProfiler. Mean ± SD below is the mean of three run means ± sample standard deviation between runs. No engine rebuild. CUDA Graph, one inference stream, no H2D/D2H in timed loop, spin wait enabled. Warmup 1000 ms; normal benchmark and separate profiling each run for at least 15 seconds. Clocks were not locked.

**trtexec exports per-layer mean/median, not per-layer P95.** Normal GPU Compute Time is measured in the unprofiled benchmark pass; per-layer statistics come from the instrumented profile pass.

## KITTI

227 matching layer names; Top 10 overlap **9/10**. Per-run overlap: [9, 9, 8]. Instrumented layer sum: 3.6649 ms.

| Run | Normal GPU mean (ms) | Median (ms) | P95 (ms) | Profile iterations |
|---:|---:|---:|---:|---:|
| 1 | 2.3502 | 2.2988 | 2.8935 | 3984 |
| 2 | 2.4247 | 2.3711 | 2.9881 | 3917 |
| 3 | 2.7566 | 2.7022 | 3.3311 | 3547 |

| trtexec rank | Python rank | Layer | Python mean (ms) | trtexec mean ± SD (ms) | Layer share (%) |
|---:|---:|---|---:|---:|---:|
| 1 | 1 | /model.22/cv2.0/cv2.0.0/conv/Conv \|\| /model.22/cv3.0/cv3.0.0/conv/Conv | 0.0727 | 0.0797 ± 0.0174 | 2.17 |
| 2 | 2 | /model.7/conv/Conv | 0.0617 | 0.0671 ± 0.0160 | 1.83 |
| 3 | 4 | /model.22/cv2.1/cv2.1.0/conv/Conv \|\| /model.22/cv3.1/cv3.1.0/conv/Conv | 0.0589 | 0.0659 ± 0.0167 | 1.80 |
| 4 | 3 | /model.22/cv2.0/cv2.0.1/conv/Conv | 0.0592 | 0.0626 ± 0.0117 | 1.71 |
| 5 | 5 | /model.1/conv/Conv | 0.0583 | 0.0620 ± 0.0095 | 1.69 |
| 6 | 7 | /model.22/cv2.2/cv2.2.0/conv/Conv \|\| /model.22/cv3.2/cv3.2.0/conv/Conv | 0.0558 | 0.0612 ± 0.0133 | 1.67 |
| 7 | 8 | /model.22/cv3.0/cv3.0.1/conv/Conv | 0.0555 | 0.0604 ± 0.0113 | 1.65 |
| 8 | 9 | /model.19/conv/Conv | 0.0530 | 0.0576 ± 0.0112 | 1.57 |
| 9 | 6 | __myl_MoveNegExpAddDivMul_myl5_0 | 0.0563 | 0.0553 ± 0.0000 | 1.51 |
| 10 | 11 | /model.22/cv2.2/cv2.2.1/conv/Conv | 0.0526 | 0.0545 ± 0.0057 | 1.49 |
| 11 | 10 | /model.0/conv/Conv | 0.0529 | 0.0542 ± 0.0014 | 1.48 |
| 12 | 13 | /model.6/m.0/cv1/conv/Conv | 0.0469 | 0.0535 ± 0.0061 | 1.46 |
| 13 | 12 | Reformatting CopyNode for Input Tensor 0 to /model.0/conv/Conv | 0.0496 | 0.0498 ± 0.0064 | 1.36 |
| 14 | 15 | /model.21/cv2/conv/Conv | 0.0420 | 0.0462 ± 0.0082 | 1.26 |
| 15 | 18 | /model.12/cv2/conv/Conv | 0.0393 | 0.0460 ± 0.0045 | 1.25 |
| 16 | 16 | /model.4/m.0/cv1/conv/Conv | 0.0411 | 0.0437 ± 0.0129 | 1.19 |
| 17 | 111 | __myl_MoveNegExpAddDivMulAdd_myl34_0 | 0.0113 | 0.0421 ± 0.0100 | 1.15 |
| 18 | 19 | /model.4/m.1/cv1/conv/Conv | 0.0381 | 0.0411 ± 0.0083 | 1.12 |
| 19 | 21 | /model.12/m.0/cv2/conv/Conv | 0.0368 | 0.0408 ± 0.0084 | 1.11 |
| 20 | 20 | /model.15/cv2/conv/Conv | 0.0369 | 0.0397 ± 0.0038 | 1.08 |

Entered Top 10: ['/model.22/cv2.2/cv2.2.1/conv/Conv']
Left Top 10: ['/model.0/conv/Conv']

## VOC

242 matching layer names; Top 10 overlap **8/10**. Per-run overlap: [7, 10, 9]. Instrumented layer sum: 3.9499 ms.

| Run | Normal GPU mean (ms) | Median (ms) | P95 (ms) | Profile iterations |
|---:|---:|---:|---:|---:|
| 1 | 2.3924 | 2.3418 | 2.9481 | 3987 |
| 2 | 2.5012 | 2.4336 | 3.0752 | 3770 |
| 3 | 3.2129 | 3.1357 | 3.8717 | 3390 |

| trtexec rank | Python rank | Layer | Python mean (ms) | trtexec mean ± SD (ms) | Layer share (%) |
|---:|---:|---|---:|---:|---:|
| 1 | 19 | __myl_MoveNegExpAddDivMul_myl134_0 | 0.0399 | 0.1147 ± 0.1684 | 2.90 |
| 2 | 1 | /model.19/conv/Conv | 0.0823 | 0.0920 ± 0.0111 | 2.33 |
| 3 | 2 | /model.22/cv2.0/cv2.0.0/conv/Conv \|\| /model.22/cv3.0/cv3.0.0/conv/Conv | 0.0717 | 0.0910 ± 0.0239 | 2.30 |
| 4 | 149 | __myl_MoveNegExpAddDivMul_myl166_0 | 0.0061 | 0.0761 ± 0.1212 | 1.93 |
| 5 | 4 | /model.7/conv/Conv | 0.0587 | 0.0750 ± 0.0196 | 1.90 |
| 6 | 5 | /model.22/cv2.1/cv2.1.0/conv/Conv \|\| /model.22/cv3.1/cv3.1.0/conv/Conv | 0.0584 | 0.0749 ± 0.0199 | 1.90 |
| 7 | 3 | /model.22/cv2.0/cv2.0.1/conv/Conv | 0.0607 | 0.0719 ± 0.0143 | 1.82 |
| 8 | 9 | /model.22/cv2.2/cv2.2.0/conv/Conv \|\| /model.22/cv3.2/cv3.2.0/conv/Conv | 0.0552 | 0.0682 ± 0.0169 | 1.73 |
| 9 | 6 | /model.1/conv/Conv | 0.0573 | 0.0669 ± 0.0124 | 1.69 |
| 10 | 7 | /model.21/cv2/conv/Conv | 0.0568 | 0.0666 ± 0.0127 | 1.69 |
| 11 | 10 | /model.22/cv3.0/cv3.0.1/conv/Conv | 0.0542 | 0.0663 ± 0.0143 | 1.68 |
| 12 | 12 | /model.22/cv2.2/cv2.2.1/conv/Conv | 0.0491 | 0.0565 ± 0.0065 | 1.43 |
| 13 | 8 | __myl_MoveNegExpAddDivMul_myl5_0 | 0.0563 | 0.0563 ± 0.0000 | 1.43 |
| 14 | 11 | /model.0/conv/Conv | 0.0530 | 0.0556 ± 0.0031 | 1.41 |
| 15 | 13 | Reformatting CopyNode for Input Tensor 0 to /model.0/conv/Conv | 0.0468 | 0.0536 ± 0.0097 | 1.36 |
| 16 | 24 | /model.12/cv2/conv/Conv | 0.0368 | 0.0481 ± 0.0137 | 1.22 |
| 17 | 17 | /model.8/cv2/conv/Conv | 0.0433 | 0.0471 ± 0.0103 | 1.19 |
| 18 | 21 | /model.21/m.0/cv1/conv/Conv | 0.0380 | 0.0456 ± 0.0072 | 1.15 |
| 19 | 25 | /model.22/dfl/Reshape + /model.22/dfl/Transpose | 0.0361 | 0.0447 ± 0.0058 | 1.13 |
| 20 | 33 | /model.8/m.0/cv1/conv/Conv | 0.0327 | 0.0410 ± 0.0083 | 1.04 |

Entered Top 10: ['__myl_MoveNegExpAddDivMul_myl134_0', '__myl_MoveNegExpAddDivMul_myl166_0']
Left Top 10: ['/model.22/cv3.0/cv3.0.1/conv/Conv', '__myl_MoveNegExpAddDivMul_myl5_0']
