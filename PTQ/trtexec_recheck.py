"""Cross-check existing TensorRT IProfiler results with the matching trtexec CLI."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'PTQ/profiling_results'
OUT = RESULTS / 'trtexec'
TRTEXEC = Path(os.environ.get('TRTEXEC_PATH',
    str(Path(os.environ['LOCALAPPDATA']) / 'NVIDIA/TensorRT-11.3-tools/bin/trtexec.exe')))
ENGINES = {'kitti':ROOT/'original/yolov8n_kitti/8n_float32_fp32.engine',
           'voc':ROOT/'original/yolov8n_voc/voc_8n_best_fp32.engine'}


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare(key):
    import cv2
    import numpy as np
    import torch
    from ultralytics.data.augment import LetterBox
    OUT.mkdir(exist_ok=True)
    source=ENGINES[key]
    data=source.read_bytes()
    size=int.from_bytes(data[:4],'little')
    metadata=json.loads(data[4:4+size])
    plan=OUT/f'{key}.plan'
    plan.write_bytes(data[4+size:])
    old=json.loads((RESULTS/f'{key}_stages.json').read_text(encoding='utf-8'))
    if hashlib.sha256(data).hexdigest()!=old['engine_sha256']:
        raise RuntimeError('Engine has changed since the previous profiling run.')
    image_path=old['samples'][0]['image']
    image=cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    # Match ProfilePredictor's actual uint8 H2D + GPU preprocessing sequence.
    array=LetterBox((640,640),auto=False)(image=image)
    tensor=torch.from_numpy(array).unsqueeze(0).to('cuda:0')
    tensor=tensor.permute(0,3,1,2).flip(1).contiguous().float().div_(255)
    values=tensor.cpu().numpy()
    input_file=OUT/f'{key}_images_fp32.bin'
    values.tofile(input_file)
    save(OUT/f'{key}_input_metadata.json',dict(engine=str(source),engine_sha256=hashlib.sha256(data).hexdigest(),
         plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),metadata=metadata,image=image_path,
         input_name='images',shape=list(values.shape),dtype=str(values.dtype),
         input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
         preparation='Same fixed representative image and GPU preprocessing as original IProfiler run.'))
    print('Prepared',key,plan,input_file,flush=True)


class Telemetry:
    def __init__(self):
        import pynvml as nv
        nv.nvmlInit()
        self.nv=nv
        self.gpu=nv.nvmlDeviceGetHandleByIndex(0)
        self.rows=[]
        self.stop_event=threading.Event()
    def poll(self):
        while not self.stop_event.is_set():
            row={'wall_time':time.time()}
            for name,func in [
                ('temperature_C',lambda:self.nv.nvmlDeviceGetTemperature(self.gpu,self.nv.NVML_TEMPERATURE_GPU)),
                ('SM_clock_MHz',lambda:self.nv.nvmlDeviceGetClockInfo(self.gpu,self.nv.NVML_CLOCK_SM)),
                ('memory_clock_MHz',lambda:self.nv.nvmlDeviceGetClockInfo(self.gpu,self.nv.NVML_CLOCK_MEM)),
                ('power_W',lambda:self.nv.nvmlDeviceGetPowerUsage(self.gpu)/1000),
                ('GPU_util_pct',lambda:self.nv.nvmlDeviceGetUtilizationRates(self.gpu).gpu)]:
                try: row[name]=func()
                except self.nv.NVMLError: row[name]=None
            self.rows.append(row)
            self.stop_event.wait(.25)
    def start(self):
        self.thread=threading.Thread(target=self.poll,daemon=True)
        self.thread.start()
        return self
    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.nv.nvmlShutdown()


def cooldown(target_C=65, timeout_s=300):
    import pynvml as nv
    nv.nvmlInit()
    gpu=nv.nvmlDeviceGetHandleByIndex(0)
    started=time.monotonic()
    try:
        while True:
            temperature=nv.nvmlDeviceGetTemperature(gpu,nv.NVML_TEMPERATURE_GPU)
            if temperature<=target_C:
                print(f'GPU cooled to {temperature} C',flush=True)
                return temperature
            if time.monotonic()-started>timeout_s:
                raise RuntimeError(f'GPU did not cool to {target_C} C; current {temperature} C')
            # Report progress periodically; this wait only runs in a child process.
            print(f'Cooling GPU: {temperature} C (target <= {target_C} C)',flush=True)
            time.sleep(10)
    finally:
        nv.nvmlShutdown()


def run(key, repeat, duration=15, family='', no_graph=False, iterations=200):
    OUT.mkdir(exist_ok=True)
    temperature_start=cooldown() if family else None
    prefix=OUT/f'{key}_{family+"_" if family else ""}run{repeat}'
    command=[str(TRTEXEC),f'--loadEngine={OUT/key}.plan',
             f'--loadInputs=images:{(OUT/f"{key}_images_fp32.bin").relative_to(ROOT).as_posix()}',
             '--warmUp=1000',f'--duration={duration}',f'--iterations={iterations}',
             '--infStreams=1','--percentile=50,95','--dumpProfile',
             f'--exportProfile={prefix}_profile.json',f'--exportTimes={prefix}_times.json']
    if no_graph:
        command.append('--noCudaGraph')
    # TensorRT 11.3 enables CUDA Graph, no data transfers, spin wait,
    # and a separate profile pass by default (verified with --help and run log).
    # Load the exact installed TensorRT libraries used by the Python baseline.
    site=Path(sys.prefix)/'Lib/site-packages'
    env=os.environ.copy()
    env['PATH']=str(site/'tensorrt_libs')+';'+str(site/'torch/lib')+';'+env['PATH']
    print('Running',key,'repeat',repeat,flush=True)
    telemetry=Telemetry().start()
    started=time.time()
    try:
        with Path(f'{prefix}.log').open('w',encoding='utf-8') as log:
            process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    finally:
        telemetry.stop()
    save(Path(f'{prefix}_run_metadata.json'),dict(command=command,started_wall_time=started,
         cooldown_target_C=65 if family else None,temperature_start_C=temperature_start,
         duration_s=time.time()-started,returncode=process.returncode,telemetry=telemetry.rows,
         note='Telemetry covers warmup + normal benchmark + separate profiling, not isolated steady-state intervals.'))
    if process.returncode:
        raise RuntimeError(Path(f'{prefix}.log').read_text(encoding='utf-8')[-8000:])
    for line in Path(f'{prefix}.log').read_text(encoding='utf-8').splitlines():
        if 'GPU Compute Time:' in line or 'Throughput:' in line or 'PASSED' in line:
            print(line,flush=True)
    print('Saved',prefix,flush=True)


def analyze(repeats=3, family='cooled', no_graph=False, duration=15, iterations=200, output_name='comparison'):
    """Equal-weight average of run means; SD measures between-run variation."""
    import numpy as np
    mode='CUDA Graph disabled' if no_graph else 'CUDA Graph enabled'
    result={'method':f'TensorRT 11.3.0.99 trtexec, original plan, real fixed input, batch 1, {mode}',
            'settings':{'warmup_ms':1000,'duration_s':duration,'minimum_iterations':iterations,'infStreams':1,
                        'cooldown_target_C':65,'run_family':family,
                        'data_transfers':False,'spin_wait':True,'separate_profile_run':True},
            'aggregation':'Equal-weight mean of per-run layer means, sample SD across runs; shared layer names.',
            'notes':['No rebuild: plan is the original engine with only the Ultralytics header removed.',
                     f'IProfiler reference: 100 synchronous graph replays; trtexec: at least {iterations} pipelined iterations per pass.',
                     'Graph mode can differ from Python reference: use the explicit settings in this report.',
                     'Per-layer P95 is NOT present in trtexec exportProfile; do not substitute median for P95.',
                     'Layer shares use the sum of instrumented layer means, not normal GPU/E2E latency.',
                     'GPU clocks/temperature are not locked. Normal benchmark and profile passes are separate.'],
            'models':{}}
    markdown=['# trtexec Top 10 cross-check','',result['method'], '',
              'Three alternating runs per model, same fixed input as Python IProfiler. '
              'Mean ± SD below is the mean of three run means ± sample standard deviation between runs. '
              f'No engine rebuild. {mode}, one inference stream, no H2D/D2H in timed loop, spin wait enabled. '
              f'Warmup 1000 ms; normal benchmark and separate profiling each run for at least {duration} seconds and {iterations} iterations. '
              'Each run starts at <=65 C after cooldown. Clocks were not locked.', '',
              '**trtexec exports per-layer mean/median, not per-layer P95.** '
              'Normal GPU Compute Time is measured in the unprofiled benchmark pass; '
              'per-layer statistics come from the instrumented profile pass.','']
    for key in ENGINES:
        reference=json.loads((RESULTS/f'{key}_layers.json').read_text(encoding='utf-8'))
        old={row['layer']:dict(row,rank=i+1) for i,row in enumerate(reference['layers'])}
        profiles=[]
        benchmarks=[]
        for repeat in range(1,repeats+1):
            prefix=OUT/f'{key}_{family+"_" if family else ""}run{repeat}'
            profile=json.loads(Path(f'{prefix}_profile.json').read_text(encoding='utf-8'))
            profiles.append({row['name']:row for row in profile if 'name' in row})
            times=json.loads(Path(f'{prefix}_times.json').read_text(encoding='utf-8'))
            compute=np.array([row['computeMs'] for row in times])
            run_meta=json.loads(Path(f'{prefix}_run_metadata.json').read_text(encoding='utf-8'))
            telemetry={}
            for metric in ['temperature_C','SM_clock_MHz','memory_clock_MHz','power_W']:
                values=[row[metric] for row in run_meta['telemetry'] if row.get(metric) is not None]
                telemetry[metric]={'mean':float(np.mean(values)),'min':min(values),'max':max(values)} if values else None
            benchmarks.append({'run':repeat,'benchmark_iterations':len(times),'profile_iterations':profile[0]['count'],
                               'GPU_compute_mean_ms':float(compute.mean()),'GPU_compute_median_ms':float(np.median(compute)),
                               'GPU_compute_P95_ms':float(np.percentile(compute,95)),
                               'whole_process_telemetry':telemetry})
        names=set(profiles[0])
        if any(set(profile)!=names for profile in profiles) or names!=set(old):
            raise RuntimeError(f'{key}: layer names do not match; cannot compare directly')
        rows=[]
        for name in names:
            values=[profile[name]['averageMs'] for profile in profiles]
            row={'layer':name,'python_rank':old[name]['rank'],'python_mean_ms':old[name]['mean_ms'],
                 'trtexec_mean_ms':float(np.mean(values)),
                 'trtexec_between_run_SD_ms':float(np.std(values,ddof=1)),
                 'trtexec_run_mean_ms':values,'trtexec_run_median_ms':[p[name]['medianMs'] for p in profiles]}
            row['run_variation_flag']=bool(min(values)>0 and max(values)/min(values)>2)
            row['mean_change_pct']=(row['trtexec_mean_ms']/row['python_mean_ms']-1)*100 if row['python_mean_ms'] else None
            rows.append(row)
        rows.sort(key=lambda row:row['trtexec_mean_ms'],reverse=True)
        total=sum(row['trtexec_mean_ms'] for row in rows)
        cumulative=0
        for i,row in enumerate(rows):
            row['trtexec_rank']=i+1
            row['layer_share_pct']=row['trtexec_mean_ms']/total*100
            cumulative+=row['layer_share_pct']
            row['cumulative_share_pct']=cumulative
        top10={row['layer'] for row in rows[:10]}
        previous_top10={row['layer'] for row in reference['layers'][:10]}
        overlap=len(top10&previous_top10)
        ranked_by_name={row['layer']:row for row in rows}
        per_run_overlap=[]
        for profile in profiles:
            per_run_top10={r['name'] for r in sorted(profile.values(),key=lambda r:r['averageMs'],reverse=True)[:10]}
            per_run_overlap.append(len(previous_top10&per_run_top10))
        model={'input_metadata':json.loads((OUT/f'{key}_input_metadata.json').read_text(encoding='utf-8')),
               'layer_count':len(rows),'repeat_count':repeats,'normal_benchmarks':benchmarks,
               'normal_GPU_compute_mean_of_run_means_ms':float(np.mean([r['GPU_compute_mean_ms'] for r in benchmarks])),
               'normal_GPU_compute_between_run_SD_ms':float(np.std([r['GPU_compute_mean_ms'] for r in benchmarks],ddof=1)),
               'profile_layer_sum_mean_ms':total,'Top10_overlap_with_python':overlap,
               'per_run_Top10_overlap_with_python':per_run_overlap,
               'entered_Top10':sorted(top10-previous_top10),'left_Top10':sorted(previous_top10-top10),
               'layers':rows,'previous_Top10_comparison':[ranked_by_name[r['layer']] for r in reference['layers'][:10]],
               'unstable_layer_names':[r['layer'] for r in rows if r['run_variation_flag']]}
        result['models'][key]=model
        save(OUT/f'{key}_comparison.json',model)
        for suffix,export_rows in [('top20',rows[:20]),('previous_top10',model['previous_Top10_comparison'])]:
            fields=['trtexec_rank','python_rank','layer','python_mean_ms','trtexec_mean_ms',
                    'trtexec_between_run_SD_ms','mean_change_pct','layer_share_pct','cumulative_share_pct','run_variation_flag']
            with (OUT/f'{key}_{suffix}.csv').open('w',encoding='utf-8-sig',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
                writer.writeheader()
                writer.writerows(export_rows)
        markdown += [f'## {key.upper()}','',
                     f"{len(rows)} matching layer names; Top 10 overlap **{overlap}/10**. "
                     f"Per-run overlap: {per_run_overlap}. Instrumented layer sum: {total:.4f} ms.", '',
                     '| Run | Normal GPU mean (ms) | Median (ms) | P95 (ms) | Profile iterations |',
                     '|---:|---:|---:|---:|---:|']
        for row in benchmarks:
            markdown.append(f"| {row['run']} | {row['GPU_compute_mean_ms']:.4f} | {row['GPU_compute_median_ms']:.4f} | {row['GPU_compute_P95_ms']:.4f} | {row['profile_iterations']} |")
        markdown += ['', '| trtexec rank | Python rank | Layer | Python mean (ms) | trtexec mean ± SD (ms) | Layer share (%) | Stability |',
                     '|---:|---:|---|---:|---:|---:|---|']
        for row in rows[:20]:
            name=row['layer'].replace('|','\\|')
            flag='>2x between-run variation; needs validation' if row['run_variation_flag'] else ''
            markdown.append(f"| {row['trtexec_rank']} | {row['python_rank']} | {name} | {row['python_mean_ms']:.4f} | {row['trtexec_mean_ms']:.4f} ± {row['trtexec_between_run_SD_ms']:.4f} | {row['layer_share_pct']:.2f} | {flag} |")
        markdown += ['',f"Entered Top 10: {model['entered_Top10']}",f"Left Top 10: {model['left_Top10']}",'']
    save(OUT/f'{output_name}.json',result)
    (OUT/f'{output_name}.md').write_text('\n'.join(markdown),encoding='utf-8')
    return result


def show_comparison(output_name='comparison'):
    from html import escape
    from IPython.display import HTML,display
    result=json.loads((OUT/f'{output_name}.json').read_text(encoding='utf-8'))
    for key,model in result['models'].items():
        print(key.upper(),f"Top 10 overlap: {model['Top10_overlap_with_python']}/10",
              '| per-layer P95 not exported by trtexec')
        html='<table><tr><th>trtexec rank</th><th>Python rank</th><th>Layer</th><th>Python mean (ms)</th><th>trtexec mean ± SD (ms)</th><th>Share (%)</th><th>Stability</th></tr>'
        for row in model['layers'][:20]:
            flag='>2x variation: inspect raw runs' if row['run_variation_flag'] else ''
            html+=f"<tr><td>{row['trtexec_rank']}</td><td>{row['python_rank']}</td><td>{escape(row['layer'])}</td><td>{row['python_mean_ms']:.4f}</td><td>{row['trtexec_mean_ms']:.4f} ± {row['trtexec_between_run_SD_ms']:.4f}</td><td>{row['layer_share_pct']:.2f}</td><td>{flag}</td></tr>"
        display(HTML(html+'</table>'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--model',choices=list(ENGINES))
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--duration',type=int,default=15)
    parser.add_argument('--family',default='cooled')
    parser.add_argument('--no-graph',action='store_true')
    parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        for key in ENGINES:
            prepare(key)
    elif args.analyze:
        analyze(family=args.family)
    else:
        if args.model is None:
            parser.error('--model is required for a run')
        run(args.model,args.repeat,args.duration,args.family,args.no_graph)
