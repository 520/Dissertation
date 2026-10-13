"""Score exported convolution output channels using weight norms and detection-loss gradients.

TensorRT plans have no autograd graph. This program checks the matching fused
PyTorch weights against ONNX and verifies predictions against the existing plan.
It does not prune, train, rebuild, or overwrite the supplied model files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn
import ultralytics
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.loss import v8DetectionLoss

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'original/yolov8n_kitti'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = payload
    if isinstance(payload, dict):
        model = payload.get('ema') if isinstance(payload.get('ema'), nn.Module) else payload['model']
    model = model.float().eval()
    saved_args = model.args
    hyp = get_cfg()
    if isinstance(saved_args, dict):
        for name, value in saved_args.items():
            if hasattr(hyp, name):
                setattr(hyp, name, value)
    model.args = hyp
    model.fuse(verbose=False)
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    # Fixed DFL projection is not a learned/prunable convolution.
    for name, parameter in model.named_parameters():
        if '.dfl.' in name:
            parameter.requires_grad_(False)
    return model


def check_onnx(model, path):
    graph = onnx.load(path)
    arrays = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    nodes_by_weight = {node.input[1]: node.name for node in graph.graph.node if node.op_type == 'Conv'}
    checks = []
    convs = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    for name, module in convs:
        for suffix, tensor in [('weight', module.weight), ('bias', module.bias)]:
            if tensor is None:
                continue
            key = f'{name}.{suffix}'
            if key not in arrays:
                raise RuntimeError(f'ONNX initializer missing: {key}')
            actual = tensor.detach().cpu().numpy()
            reference = arrays[key]
            if actual.shape != reference.shape:
                raise RuntimeError(f'ONNX shape differs for {key}')
            error = float(np.max(np.abs(actual - reference)))
            if not np.allclose(actual, reference, rtol=1e-5, atol=1e-6):
                raise RuntimeError(f'Checkpoint differs from exported ONNX: {key}, max error={error}')
            checks.append({'parameter': key, 'max_abs_error': error, 'matched': True})
    return convs, nodes_by_weight, checks


def dataset_at(path, size):
    data = {'path': str(path), 'train': 'images/train', 'val': 'images/val',
            'names': {0:'car',1:'van',2:'truck',3:'pedestrian',4:'person_sitting',5:'cyclist',6:'tram',7:'misc'},
            'nc': 8}
    dataset = YOLODataset(img_path=str(path / 'images/train'), data=data, imgsz=size,
                          augment=False, rect=False, cache=False, stride=32,
                          batch_size=1, hyp=get_cfg(), prefix='Channel scoring: ')
    files = sorted((path / 'images/train').glob('*.png'))
    if set(map(str, files)) != set(dataset.im_files):
        raise RuntimeError('Dataset loader omitted or changed the training image list')
    for image in files:
        if not (path / 'labels/train' / (image.stem + '.txt')).is_file():
            raise RuntimeError(f'Missing label: {image}')
    return dataset


def batch_on_device(raw, device):
    return {'img': raw['img'].to(device, non_blocking=True).float() / 255,
            'cls': raw['cls'].to(device), 'bboxes': raw['bboxes'].to(device),
            'batch_idx': raw['batch_idx'].to(device)}


def verify_engine(model, dataset, engine_path, output_dir, device):
    import tensorrt as trt
    raw = engine_path.read_bytes()
    length = int.from_bytes(raw[:4], 'little')
    metadata = {}
    if 0 < length < 1024 * 1024:
        try:
            metadata = json.loads(raw[4:4+length])
            raw = raw[4+length:]
        except (ValueError, UnicodeDecodeError):
            pass
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(raw)
    if engine is None:
        raise RuntimeError('Could not deserialize the existing TensorRT plan')
    context = engine.create_execution_context()
    inspector = engine.create_engine_inspector()
    engine_layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    write_json(output_dir / 'engine_inspector.json', engine_layers)
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    outputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError('Expected one input and one output')
    checks = []
    model.eval()
    for index in [0, len(dataset)//2, len(dataset)-1]:
        batch = batch_on_device(YOLODataset.collate_fn([dataset[index]]), device)
        x = batch['img'].contiguous()
        if tuple(x.shape) != tuple(engine.get_tensor_shape(inputs[0])):
            raise RuntimeError('Input size differs from the existing engine')
        output = torch.empty(tuple(engine.get_tensor_shape(outputs[0])), device=device, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(x)
            prediction = prediction[0] if isinstance(prediction, tuple) else prediction
            context.set_tensor_address(inputs[0], x.data_ptr())
            context.set_tensor_address(outputs[0], output.data_ptr())
            if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError('TensorRT inference failed')
            torch.cuda.synchronize()
            difference = (prediction-output).abs()
            old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
            old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                tf32_prediction = model(x)
                tf32_prediction = tf32_prediction[0] if isinstance(tf32_prediction, tuple) else tf32_prediction
                tf32_difference = (tf32_prediction-output).abs()
            finally:
                torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
                torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            # Stored FP32 plans can use TF32 tactics. Record numerical error;
            # output consistency is an empirical check, not binary provenance.
            passed = bool(torch.allclose(prediction, output, rtol=5e-3, atol=5e-3))
            record = {'image':dataset.im_files[index], 'max_abs_error':float(difference.max()),
                      'mean_abs_error':float(difference.mean()),
                      'box_max_abs_error':float(difference[:,:4].max()),
                      'class_max_abs_error':float(difference[:,4:].max()),
                      'tf32_pytorch_max_abs_error':float(tf32_difference.max()),
                      'tf32_pytorch_mean_abs_error':float(tf32_difference.mean()),
                      'tf32_pytorch_allclose':bool(torch.allclose(tf32_prediction,output,rtol=5e-3,atol=5e-3)),
                      'rtol':5e-3, 'atol':5e-3, 'passed':passed}
            checks.append(record)
            if not passed:
                print(f'Engine strict output comparison differs: {record}',flush=True)
    return {'tensorrt':trt.__version__, 'engine_metadata':metadata,
            'prediction_checks':checks,
            'strict_output_match':all(record['passed'] for record in checks),
            'source_attribution':'ONNX-matched source model; engine internal weights not recoverable; no claim of exact engine-gradient equivalence',
            'inspector_layers':engine_layers.get('Layers', [])}


def add_ranks(rows, columns, prefix):
    for column in columns:
        ranked = sorted((row for row in rows if row[column] is not None),
                        key=lambda row: (-row[column], row.get('layer_index', 0), row.get('channel', 0)))
        last_value, rank = None, 0
        for index, row in enumerate(ranked, 1):
            if row[column] != last_value:
                rank = index
                last_value = row[column]
            row[f'{prefix}_{column}'] = rank
        for row in rows:
            row.setdefault(f'{prefix}_{column}', None)


def export_results(output_dir, convs, state, manifest, nodes_by_weight):
    channels, layers = [], []
    columns = ['l1', 'l2', 'taylor1']
    engine_names = manifest['engine_validation']['inspector_layers']
    for index, (name, conv) in enumerate(convs):
        weights = conv.weight.detach().double().cpu().flatten(1)
        l1 = weights.abs().sum(1).numpy()
        l2 = weights.square().sum(1).sqrt().numpy()
        count = state['counts'][name]
        taylor = state['sums'][name].cpu().numpy() / count if count else None
        weight_only = state['weight_sums'][name].cpu().numpy() / count if count else None
        node = nodes_by_weight.get(name+'.weight', '')
        matching = [str(layer.get('Name', '') if isinstance(layer, dict) else layer)
                    for layer in engine_names]
        matching = [layer for layer in matching if node and node in layer and not layer.startswith('Reformatting')]
        fixed = '.dfl.' in name
        rows = []
        for channel in range(conv.out_channels):
            rows.append({'layer_index':index, 'layer':name, 'onnx_node':node,
                         'engine_layers':' | '.join(matching), 'channel':channel,
                         'role':'fixed_dfl_projection' if fixed else 'learned_conv',
                         'pruning_candidate':not fixed and not (name.startswith('model.22.') and name.endswith('.2')),
                         'in_channels':conv.in_channels, 'groups':conv.groups,
                         'kernel':f'{conv.kernel_size[0]}x{conv.kernel_size[1]}',
                         'weights_per_channel':weights.shape[1], 'l1':float(l1[channel]),
                         'l2':float(l2[channel]),
                         'l1_mean_abs':float(l1[channel]/weights.shape[1]),
                         'l2_rms':float(l2[channel]/math.sqrt(weights.shape[1])),
                         'taylor1':float(taylor[channel]) if taylor is not None else None,
                         'taylor1_weight_only':float(weight_only[channel]) if weight_only is not None else None,
                         'gradient_images':count,
                         'taylor_status':'MEASURED' if count else 'NOT_IN_DETECTION_LOSS_GRAPH'})
        add_ranks(rows, columns, 'rank_within_layer')
        channels.extend(rows)
        layers.append({'layer_index':index, 'layer':name, 'out_channels':conv.out_channels,
                       'role':'fixed_dfl_projection' if fixed else 'learned_conv',
                       'l1':float(l1.sum()), 'l2':float(np.sqrt(np.square(l2).sum())),
                       'taylor1':float(taylor.sum()) if taylor is not None else None,
                       'mean_channel_l1':float(l1.mean()), 'mean_channel_l2':float(l2.mean()),
                       'mean_channel_taylor1':float(taylor.mean()) if taylor is not None else None,
                       'min_channel_taylor1':float(taylor.min()) if taylor is not None else None,
                       'max_channel_taylor1':float(taylor.max()) if taylor is not None else None,
                       'gradient_images':count})
    add_ranks(channels, columns, 'rank_global')
    add_ranks(layers, columns, 'rank_layer')
    write_csv(output_dir/'channels.csv', channels)
    write_csv(output_dir/'layers.csv', layers)
    write_json(output_dir/'scores.json', {'manifest':manifest, 'layers':layers, 'channels':channels})
    for column in columns:
        ranked = sorted((row for row in channels if row[column] is not None),
                        key=lambda row:(-row[column], row['layer_index'], row['channel']))
        write_csv(output_dir/f'channels_ranked_{column}.csv', ranked)
        ranked_layers = sorted((row for row in layers if row[column] is not None),
                               key=lambda row:(-row[column], row['layer_index']))
        write_csv(output_dir/f'layers_ranked_{column}.csv', ranked_layers)
    per_layer = output_dir/'per_layer'
    per_layer.mkdir(exist_ok=True)
    for name, _ in convs:
        rows = [row for row in channels if row['layer']==name]
        rows.sort(key=lambda row:(row['taylor1'] is None,-(row['taylor1'] or 0),row['channel']))
        write_csv(per_layer/(name+'.csv'), rows)
    report = ['# KITTI YOLOv8n 通道重要性评分', '',
              f"状态：{manifest['status']}；评分图像：{manifest['processed_images']}/{manifest['total_training_images']}；卷积层：{len(convs)}；输出通道：{len(channels)}。", '',
              '## 评分定义与来源', '',
              '- 通道从 0 开始编号。排名从 1 开始，1 表示最高分；相同分数共享排名。',
              '- L1 = Σ |W[c,i,h,w]|；L2 = √Σ W[c,i,h,w]²。使用与 ONNX 对应的 BN 融合后卷积权重，不包括偏置。',
              '- Taylor-1 = 全训练图像平均 |Σ A[c,h,w] × ∂检测损失/∂A[c,h,w]|。A 是融合卷积输出、激活函数之前；每张图像独立反向传播，先取绝对值再平均。',
              '- 对融合卷积，该 Taylor-1 等价于 |Σ W×∂L/∂W + b×∂L/∂b|；另提供 taylor1_weight_only，排除偏置。',
              '- 检测损失 = 7.5×box + 0.5×classification + 1.5×DFL。仅使用训练集真实标签，640×640，batch=1，FP32，无数据增强，无优化器更新。',
              '- engine 没有训练反向图：Taylor 在权重已与 ONNX 核对的可求梯度 PyTorch 图上计算。引擎与 PyTorch 在三张真实图像上做输出对照，误差和严格比较结果见 manifest；内部权重无法恢复，因此不能宣称这就是 engine 的精确梯度。',
              '- 各 TensorRT 融合层与源卷积可能多对一，映射见 engine_layers；Concat、SiLU、Reformat 等无独立卷积核权重，不定义权重 L1/L2。',
              '- DFL 固定投影不参与该训练损失的解码路径：Taylor 标为不可用，不能解释为不重要。最终检测输出通道以及 DFL 通道不应直接剪除。',
              '- 跨层原始 L1/L2 受滤波器尺寸影响：同时给出 l1_mean_abs、l2_rms；层内排名更适合比较同一层通道。',
              '- 层级 Taylor 是各通道绝对一阶分数之和，不是同时删除整层的损失变化。分数不是实测 mAP 下降。', '',
              '## Taylor-1 层排序（前 20，降序）', '',
              '| 排名 | 源卷积 | 通道数 | 通道分数之和 | 通道平均分数 |', '|---:|---|---:|---:|---:|']
    for row in sorted((r for r in layers if r['taylor1'] is not None), key=lambda r:-r['taylor1'])[:20]:
        report.append(f"| {row['rank_layer_taylor1']} | {row['layer']} | {row['out_channels']} | {row['taylor1']:.8g} | {row['mean_channel_taylor1']:.8g} |")
    report += ['', '## 数据与文件', '',
               '- channels.csv：每层每通道的全部分数、层内排名及全局排名。',
               '- channels_ranked_l1/l2/taylor1.csv：三个指标各自的全局降序排序。',
               '- layers_ranked_l1/l2/taylor1.csv：三个指标各自的层级降序排序。',
               '- per_layer/*.csv：按源层名分文件，可按 rank_within_layer_* 排序。',
               '- manifest.json / images.csv：数据覆盖、样本哈希、版本、公式和验证信息。', '',
               '## 方法参考', '',
               '- https://arxiv.org/abs/1611.06440',
               '- https://docs.ultralytics.com/reference/utils/loss/',
               '- https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip', '']
    (output_dir/'report.md').write_text('\n'.join(report), encoding='utf-8')
    return layers, channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'PTQ/datasets/kitti')
    parser.add_argument('--output', type=Path, default=ROOT/'sensitivity_results/kitti_channel_scores')
    parser.add_argument('--limit', type=int, default=0, help='0 means all training images')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-enabled PyTorch is required')
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model_path = MODEL_DIR/'8n_float32.pt'
    onnx_path = MODEL_DIR/'8n_float32_fp32.onnx'
    engine_path = MODEL_DIR/'8n_float32_fp32.engine'
    print('Loading model and checking exported weights...', flush=True)
    model = load_model(model_path, device)
    convs, nodes_by_weight, onnx_checks = check_onnx(model, onnx_path)
    dataset = dataset_at(args.data, 640)
    print(f'Dataset: {len(dataset)} training images. Checking engine predictions...', flush=True)
    engine_validation = verify_engine(model, dataset, engine_path, args.output, device)
    paths = [Path(path) for path in dataset.im_files]
    sample_records = [{'index':index, 'image':str(path.relative_to(args.data)),
                       'image_sha256':sha256(path),
                       'label':str(Path('labels/train')/(path.stem+'.txt')),
                       'label_sha256':sha256(args.data/'labels/train'/(path.stem+'.txt'))}
                      for index,path in enumerate(paths)]
    write_csv(args.output/'images.csv', sample_records)
    manifest = {'status':'RUNNING', 'model':str(model_path), 'model_sha256':sha256(model_path),
                'onnx':str(onnx_path), 'onnx_sha256':sha256(onnx_path),
                'engine':str(engine_path), 'engine_sha256':sha256(engine_path),
                'torch':torch.__version__, 'ultralytics':ultralytics.__version__,
                'gpu':torch.cuda.get_device_name(), 'imgsz':640, 'batch':1, 'seed':42,
                'weights':'BN-fused checkpoint, matched to ONNX initializers',
                'taylor1':'mean_images(abs(sum_spatial(activation * loss_gradient))), pre-activation fused convolution output',
                'loss_gains':{key:getattr(model.args,key) for key in ('box','cls','dfl')},
                'dataset':str(args.data), 'dataset_source':'https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip',
                'dataset_archive_sha256':sha256(args.data.parent/'downloads/kitti.zip'),
                'total_training_images':len(dataset), 'processed_images':0,
                'onnx_weight_checks':onnx_checks, 'engine_validation':engine_validation,
                'gradient_check':'Fused-convolution gate derivative cross-checked against weights and bias derivative on first image',
                'rank_direction':'descending; rank 1 is highest; tied values share competition rank',
                'channel_index_base':0}
    target_count = min(args.limit, len(dataset)) if args.limit else len(dataset)
    state = {'sums':{name:torch.zeros(conv.out_channels,device=device,dtype=torch.float64) for name,conv in convs},
             'weight_sums':{name:torch.zeros(conv.out_channels,device=device,dtype=torch.float64) for name,conv in convs},
             'counts':{name:0 for name,_ in convs}, 'processed':0, 'losses':[]}
    checkpoint = args.output/'progress.pt'
    if args.resume and checkpoint.exists():
        saved = torch.load(checkpoint,map_location=device,weights_only=False)
        if saved['model_sha256']!=manifest['model_sha256'] or saved['images_sha256']!=sha256(args.output/'images.csv'):
            raise RuntimeError('Cannot resume: model or dataset has changed')
        state = saved['state']
    hooks, current_signed = [], {}
    def forward_hook(name):
        def hook(module, inputs, output):
            if output.requires_grad:
                # SiLU may operate in place: preserve the pre-activation value.
                activation = output.detach().clone()
                def capture(gradient):
                    signed = (activation*gradient.detach()).sum(dim=(0,2,3),dtype=torch.float64)
                    current_signed[name] = signed
                    state['sums'][name] += signed.abs()
                    state['counts'][name] += 1
                output.register_hook(capture)
        return hook
    for name, conv in convs:
        hooks.append(conv.register_forward_hook(forward_hook(name)))
    criterion = v8DetectionLoss(model)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,range(state['processed'],target_count)),
                                         batch_size=1,shuffle=False,num_workers=args.workers,pin_memory=True,
                                         collate_fn=YOLODataset.collate_fn)
    model.eval()
    # Raw detection outputs are required for the training loss, but BN/other
    # modules stay in evaluation mode and no training update is performed.
    model.model[-1].training = True
    begin = time.perf_counter()
    initial_count = state['processed']
    def save_progress():
        manifest['processed_images'] = state['processed']
        manifest['elapsed_scoring_seconds'] = time.perf_counter()-begin
        write_json(args.output/'manifest.json',manifest)
        cpu_state = {**state, 'sums':{k:v.cpu() for k,v in state['sums'].items()},
                     'weight_sums':{k:v.cpu() for k,v in state['weight_sums'].items()}}
        torch.save({'state':cpu_state,'model_sha256':manifest['model_sha256'],
                    'images_sha256':sha256(args.output/'images.csv')},checkpoint)
    print(f'Scoring {target_count} images; each image has a separate backward pass.',flush=True)
    try:
        for raw in loader:
            current_signed.clear()
            model.zero_grad(set_to_none=True)
            batch = batch_on_device(raw,device)
            loss_terms, detached = criterion(model(batch['img']),batch)
            loss = loss_terms.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at image {state["processed"]}')
            loss.backward()
            for name, conv in convs:
                if name not in current_signed:
                    if '.dfl.' not in name:
                        raise RuntimeError(f'Unexpected disconnected convolution: {name}')
                    continue
                weight_signed = (conv.weight.detach().double()*conv.weight.grad.detach().double()).flatten(1).sum(1)
                state['weight_sums'][name] += weight_signed.abs()
                if state['processed']==0:
                    gate_signed = weight_signed
                    if conv.bias is not None:
                        gate_signed = gate_signed+conv.bias.detach().double()*conv.bias.grad.detach().double()
                    if not torch.allclose(gate_signed,current_signed[name],rtol=1e-3,atol=1e-4):
                        raise RuntimeError(f'Taylor derivative consistency failed: {name}')
            state['losses'].append(float(loss.detach()))
            state['processed'] += 1
            if state['processed']==1 or state['processed']%100==0 or state['processed']==target_count:
                seconds=time.perf_counter()-begin
                rate=(state['processed']-initial_count)/seconds
                print(f"PROGRESS {state['processed']}/{target_count}, {rate:.2f} images/s, loss={state['losses'][-1]:.5f}",flush=True)
                save_progress()
    except Exception:
        manifest['status']='FAILED'
        save_progress()
        raise
    finally:
        for hook in hooks:
            hook.remove()
    manifest['status']='COMPLETE' if state['processed']==len(dataset) else 'SUBSET_ONLY'
    manifest['processed_images']=state['processed']
    manifest['loss_summary']={'mean':float(np.mean(state['losses'])), 'min':min(state['losses']), 'max':max(state['losses'])}
    manifest['gradient_coverage']={name:count for name,count in state['counts'].items()}
    save_progress()
    write_csv(args.output/'image_losses.csv',[{'image_index':i,'image':str(paths[i]),'loss':loss} for i,loss in enumerate(state['losses'])])
    layers,channels=export_results(args.output,convs,state,manifest,nodes_by_weight)
    print(json.dumps({'status':manifest['status'],'images':state['processed'],'convolutions':len(layers),
                      'channels':len(channels),'output':str(args.output)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
