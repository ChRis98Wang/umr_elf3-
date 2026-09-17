#!/usr/bin/env python3
"""Resumable production queue over every pinned local AMASS source for ELF3.

No standalone throughput benchmark. Bounded jobs, exact source hashes, separate
attempt/completion/quality counts, no promotion or policy training. Run inside a
finite owned bfm-umr-refresh-elf3-*.service; Ctrl-C/stop kills owned job groups.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from elf3_umr_asset import digest, write_json, verify_assets

ROOT=Path(__file__).resolve().parents[1]
FILES=('run_elf3_library.py','elf3_library_worker.py','refine_elf3_whole_body.py',
       'elf3_whole_body_model.py','elf3_mesh_distance.py','refine_elf3_arm_spline.py',
       'refine_elf3_umr_trajectory.py','replay_elf3_umr_stage2.py','run_elf3_umr_trial.py',
       'elf3_umr_asset.py','elf3_source_reuse.py','diagnose_elf3_arm_motion.py',
       'umr_smplx_source.py','umr_full_source_v5.py','umr_heading_source_v6.py','umr_root_initialization_v6.py')


def atomic_json(path,value):
    path=Path(path)
    with tempfile.NamedTemporaryFile(mode='w',prefix=path.name+'.',suffix='.tmp',dir=path.parent,delete=False) as f:
        temp=Path(f.name)
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def verify_frozen(plan):
    for path,sha in plan['protected'].items():
        if digest(path)!=sha:raise ValueError('Frozen pipeline input changed: '+path)


def child(command,logfile,timeout,env=None):
    # Inherit the job's session/group, so the supervisor owns all descendants.
    with Path(logfile).open('x') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        try:return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
            return 124


def stop_group(process):
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=10)
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass


def unit_guard():
    found=re.findall(r'(?:^|/)(bfm-umr-refresh-elf3-[A-Za-z0-9_-]+\.service)(?:/|$)',
                     Path('/proc/self/cgroup').read_text(),re.MULTILINE)
    if len(found)!=1:raise RuntimeError('Owned finite bfm-umr-refresh-elf3-*.service required')
    out=subprocess.check_output(['systemctl','--user','show',found[0],'-p','KillMode','-p','Restart',
        '-p','RuntimeMaxUSec','-p','TimeoutStopUSec','-p','MemoryMax','-p','TasksMax'],text=True,timeout=10)
    props=dict(line.split('=',1) for line in out.splitlines())
    if (props.get('KillMode')!='control-group' or props.get('Restart')!='no'
            or any(props.get(k) in (None,'','0','infinity') for k in
                   ('RuntimeMaxUSec','TimeoutStopUSec','MemoryMax','TasksMax'))):
        raise RuntimeError('Finite resource limits and entire-group cleanup required')
    return found[0]


def ordered_jobs(batch):
    if batch.get('schema')!='bfm.elf3_batch_plan/1':raise ValueError('Wrong library schema')
    jobs=batch['jobs'];seen=set()
    for row in jobs:
        sha=row['source_sha256']
        if (not re.fullmatch('[0-9a-f]{64}',sha) or sha in seen or row['split'] not in ('train','validation')
                or row['output_robot']!='elf3_dof31' or row['target_50hz_frames']<2):
            raise ValueError('Invalid/duplicate source or changed split/robot')
        seen.add(sha)
    if len(jobs)!=batch['unique_jobs']:raise ValueError('Library size mismatch')
    # Grouping is a scheduling hint only; actual cache reuse verifies every
    # canonical array. Every source remains in the queue, including long clips.
    return sorted(jobs,key=lambda j:(j['shape_betas_sha256'],j['origin_id']))


def verified_result(path,row):
    result=json.loads(Path(path).read_text())
    if (result['source_sha256']!=row['source_sha256'] or result['split']!=row['split']
            or result['frames']!=row['target_50hz_frames'] or result['training_approved'] is not False):
        raise ValueError('Job result does not bind complete original motion')
    for name,sha in result.get('artifacts',{}).items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unexpected artifact path')
        if digest(Path(path).parent/relative)!=sha:raise ValueError('Completed artifact changed')
    if result.get('stage2_complete') is True:
        raw=json.loads((Path(path).parent/'raw/receipt.json').read_text())
        if (raw['full_duration_verified'] is not True or raw['frames']!=row['target_50hz_frames']
                or digest(Path(path).parent/'raw/motion.npz')!=raw['motion_sha256']):
            raise ValueError('Missing complete retargeted artifact')
    return result


def counts(jobs,finished,active):
    values=list(finished.values())
    return {'total_sources':len(jobs),'finished_attempts':len(values),'active_jobs':list(active),
        'pending_sources':len(jobs)-len(values)-len(active),
        'stage2_complete':sum(r.get('stage2_complete',False) for r in values),
        'refinement_complete':sum(r.get('refinement_complete',False) for r in values),
        'by_status':dict(Counter(r['status'] for r in values)),
        'training_approved':0,'training_started':False,
        'all_library_retargeted':len(values)==len(jobs) and all(r.get('stage2_complete') for r in values)}


def run(args):
    unit=unit_guard()
    if not 1<=args.workers<=4 or not 60<=args.max_seconds<=82800:raise ValueError('Finite <=4 worker / <=23h submit budget')
    output=args.output.resolve()
    if output.is_symlink():raise ValueError('No redirected output')
    existed=output.exists()
    if existed and not args.resume:raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    # Prevent two supervisors from racing identical jobs after --resume.
    with (output/'supervisor.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assets=args.assets.resolve();verify_assets(assets)
        # SMPL-X and UMR already live in separate installed environments.
        # Validate imports before admitting ANY production source; do not install.
        subprocess.run([str(args.prepare_python.absolute()),'-c','import smplx,torch,numpy'],
                       env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True,timeout=30)
        protected_paths=[ROOT/'scripts'/name for name in FILES]+[args.batch_plan.resolve(),args.robot_xml.resolve(),
            args.body_model.resolve(),assets/'provenance.json',assets/'elf3.urdf',
            args.prepare_python.absolute(),
            ROOT/'docs/UMR_HEADING_RECOVERY_V6_20260912.md']
        protected={str(p):digest(p) for p in protected_paths}
        plan={'schema':'bfm.elf3_library_queue/1','jobs':ordered_jobs(json.loads(args.batch_plan.read_text())),
            'protected':protected,'config':{'assets':str(assets),'urdf':str(assets/'elf3.urdf'),
                'robot_xml':str(args.robot_xml.resolve()),'body_model':str(args.body_model.resolve()),
                'prepare_python':str(args.prepare_python.absolute()),
                'stage1_cache':str(output/'stage1_cache'),'workers':args.workers,
                'source_preparation':'unchanged_full_v6','full_body_iterations':450,'full_body_seconds':600,
                'long_refinement':'pending_not_cropped','automatic_training':False}}
        if (output/'plan.json').exists():
            if json.loads((output/'plan.json').read_text())!=plan:raise ValueError('Resume requires identical code, sources and recipe')
        else:write_json(output/'plan.json',plan)
        jobs=plan['jobs'];finished={};active={};stop=[False];start=time.monotonic();errors=0
        for i,row in enumerate(jobs):
            path=output/f'job_{i:05d}/completed.json'
            if path.exists():
                pointer=json.loads(path.read_text())
                receipt=Path(pointer['result'])
                if digest(receipt)!=pointer['sha256']:raise ValueError('Result pointer changed')
                finished[i]=verified_result(receipt,row)
        previous={s:signal.signal(s,lambda signum,frame:stop.__setitem__(0,True)) for s in (signal.SIGTERM,signal.SIGINT)}
        pending=iter(i for i in range(len(jobs)) if i not in finished)
        exhausted=False;pause_reason=None
        def status():
            atomic_json(output/'status.json',{'schema':'bfm.elf3_library_status/1',**counts(jobs,finished,active),
                'unit':unit,'elapsed_seconds':time.monotonic()-start,'paused_reason':pause_reason,
                'plan_sha256':digest(output/'plan.json')})
        try:
            while True:
                if stop[0]:pause_reason='stop_requested';break
                if time.monotonic()-start>args.max_seconds:exhausted=True;pause_reason='submit_time_budget'
                if shutil.disk_usage(output).free<50*(1<<30):exhausted=True;pause_reason='50_GiB_disk_reserve'
                while not exhausted and len(active)<args.workers:
                    try:i=next(pending)
                    except StopIteration:exhausted=True;break
                    folder=output/f'job_{i:05d}';folder.mkdir(exist_ok=True)
                    attempt=next(folder/f'attempt_{n:03d}' for n in range(1000)
                        if not (folder/f'attempt_{n:03d}').exists() and not (folder/f'attempt_{n:03d}.log').exists())
                    log=(folder/(attempt.name+'.log')).open('x')
                    command=[sys.executable,str(ROOT/'scripts/elf3_library_worker.py'),'--plan',str(output/'plan.json'),
                             '--index',str(i),'--output',str(attempt)]
                    p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[i]=(p,log,attempt,time.monotonic())
                    print('[start]',i,jobs[i]['origin_id'],jobs[i]['target_50hz_frames'],flush=True)
                for i,(p,log,attempt,began) in list(active.items()):
                    timed_out=time.monotonic()-began>4200
                    if p.poll() is None and not timed_out:continue
                    stop_group(p);log.close()
                    result_path=attempt/'result.json'
                    if result_path.exists():
                        result=verified_result(result_path,jobs[i]);errors=0
                    else:
                        attempt.mkdir(exist_ok=True)
                        result={'schema':'bfm.elf3_library_job/1','source_sha256':jobs[i]['source_sha256'],
                            'split':jobs[i]['split'],'origin_id':jobs[i]['origin_id'],'frames':jobs[i]['target_50hz_frames'],
                            'status':'execution_failed','reason':'job_timeout' if timed_out else f'worker_exit_{p.returncode}',
                            'stage2_complete':False,'refinement_complete':False,'training_approved':False,
                            'artifacts':{}}
                        write_json(result_path,result);errors+=1
                    write_json(attempt.parent/'completed.json',{'result':str(result_path),'sha256':digest(result_path)})
                    finished[i]=result;del active[i]
                    print('[done]',i,result['status'],flush=True)
                    if errors>=8:exhausted=True;pause_reason='8_consecutive_execution_errors'
                status()
                if exhausted and not active:break
                time.sleep(1)
        finally:
            for p,log,_,_ in active.values():stop_group(p);log.close()
            active.clear();status()
            for s,handler in previous.items():signal.signal(s,handler)
        print(json.dumps(counts(jobs,finished,{})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--robot-xml',type=Path,required=True)
    p.add_argument('--body-model',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--prepare-python',type=Path,required=True,
                   help='Python executable in your existing SMPL-X environment')
    p.add_argument('--max-seconds',type=int,default=82800);p.add_argument('--resume',action='store_true')
    run(p.parse_args())
