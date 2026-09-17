#!/usr/bin/env python3
"""One complete AMASS -> ELF3 job. No cropping, policy training or promotion.

Full source preparation is delegated to the unchanged v6/v5 adapter. Stage I
depends only on verified canonical arrays, robot geometry, recipe and runtime;
Stage II keeps ONE continuous IK state over the entire source clock.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'external/umr_trial_20260908')]
from elf3_umr_asset import digest, write_json, verify_assets
from elf3_source_reuse import load_replay_source, human_stage1_identity
from run_elf3_umr_trial import UMR, elf3_config, elf3_mesh_sole_points, motion_audit
from replay_elf3_umr_stage2 import foot_feasible_initial_pose
from umr_smplx_source import (SEGMENTS, SmplxSurfaceHuman, canonical_json,
    model_geometry_fingerprint, environment_packages, verify_umr_checkout)


@contextmanager
def lock(path):
    with Path(path).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def stage1_context(prepared, xml):
    from umr.bodies.robot import RobotBody, RobotSpec
    from umr.bodies.surface import SurfacePointCloud
    from umr.retarget.pipeline import select_correspondence_points
    cfg=elf3_config(xml,4096,2500)
    cfg['correspondence']['device']='cuda'
    robot=RobotBody(xml,RobotSpec.from_config(cfg.robot))
    human=SmplxSurfaceHuman(prepared,robot.height())
    sampled=select_correspondence_points(prepared['segment'],SEGMENTS,4096,
        points=prepared['canonical_points'],method='fps')
    ids=prepared['binding_joint_ids'][sampled]
    rotations=human.data.xmat[ids].reshape(-1,3,3)
    canonical=prepared['canonical_points'][sampled]*human.scale
    local_pos=np.einsum('nji,nj->ni',rotations,canonical-human.data.xpos[ids])
    local_normal=np.einsum('nji,nj->ni',rotations,prepared['canonical_normals'][sampled])
    cloud=SurfacePointCloud(canonical,prepared['canonical_normals'][sampled],ids,
        np.full(len(ids),-1,dtype=np.int64),local_pos,local_normal,prepared['segment'][sampled],SEGMENTS)
    return cfg,robot,human,cloud,sampled


def checked_cache(entry, identity):
    receipt=json.loads((entry/'receipt.json').read_text())
    if receipt['identity']!=identity:raise ValueError('Stage I identity changed')
    for path,sha in receipt['artifacts'].items():
        if digest(entry/path)!=sha:raise ValueError('Stage I artifact changed')
    seed=json.loads((entry/'inputs.json').read_text())
    if digest(seed['source'])!=seed['source_sha256']:raise ValueError('Stage I source changed')
    return receipt


def obtain_stage1(cache, prepared, source, cfg, robot, human, cloud, sampled, protected):
    from umr.bodies.surface import sample_model_surface
    from umr.paths import SetupLayout
    from umr.stages import learn_correspondence
    identity={'schema':'bfm.elf3_library_stage1_identity/1','human':human_stage1_identity(prepared),
        'robot_geometry':model_geometry_fingerprint(robot.model),'robot_xml_sha256':digest(cfg.robot['xml']),
        'config':cfg,'scale':human.scale,'umr_commit':verify_umr_checkout(UMR),
        'runtime':environment_packages(('numpy','scipy','mujoco','torch','mink','trimesh')),
        'worker_sha256':digest(__file__)}
    key=hashlib.sha256(canonical_json(identity)).hexdigest()
    cache.mkdir(parents=True,exist_ok=True)
    entry=cache/key
    with lock(cache/(key+'.lock')):
        if entry.exists():
            checked_cache(entry,identity)
            print('[stage1-cache-hit]',key,flush=True)
            return entry
        temp=Path(tempfile.mkdtemp(prefix=key+'.partial-',dir=cache))
        setup=SetupLayout(temp/'setup').ensure()
        inputs={'source':str(source),'source_sha256':digest(source),'robot_xml':cfg.robot['xml'],
            'robot_xml_sha256':digest(cfg.robot['xml']),'config':cfg,'robot_geometry':identity['robot_geometry'],
            'umr_commit':identity['umr_commit'],'source_metadata':prepared['metadata']}
        write_json(temp/'inputs.json',inputs)
        robot_cloud=sample_model_surface(robot.model,robot.data,len(sampled),geom_ids=robot.surface_geoms(),
            oversample=cfg.sampling['oversample'],cull_margin=cfg.sampling['cull_margin'],seed=0,segment_names=SEGMENTS)
        np.savez(setup.bodies,stamp=digest(temp/'inputs.json'),robot_xml=cfg.robot['xml'],
            **cloud.to_dict('human_'),**robot_cloud.to_dict('robot_'))
        np.savez(temp/'samples.npz',source_surface_indices=sampled)
        # One shared GPU learner at a time; raw FK/IK and preparation stay CPU.
        with lock(cache/'gpu-learning.lock'):
            import torch
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                '--format=csv,noheader,nounits','--id=0'],text=True,timeout=10).strip())
            if free<4096:raise RuntimeError('Need 4 GiB free GPU memory; no other process will be stopped')
            torch.set_num_threads(1)
            torch.cuda.set_per_process_memory_fraction(.20,0)
            torch.manual_seed(0)
            learn_correspondence(cfg,setup,epochs=2500,device='cuda')
        with np.load(setup.correspondence,allow_pickle=True) as z:
            history=z['loss_history']
            if (history.shape!=(2500,5) or not np.isfinite(history).all()
                    or not np.array_equal(history[:,0],np.arange(2500))):raise ValueError('Incomplete Stage I')
        for path,sha in protected.items():
            if digest(path)!=sha:raise ValueError('Pipeline changed during learning')
        artifacts={str(p.relative_to(temp)):digest(p) for p in temp.rglob('*') if p.is_file()}
        write_json(temp/'receipt.json',{'schema':'bfm.elf3_library_stage1/1','identity':identity,
            'artifacts':artifacts,'epochs':2500,'device':'cuda','source_is_complete':True})
        os.rename(temp,entry)
        checked_cache(entry,identity)
        return entry


def solve_full(prepared, source, stage1, output, cfg, robot, human, sampled, urdf):
    from scipy.spatial.transform import Rotation
    from umr.bodies.surface import SurfacePointCloud
    from umr.retarget.binding import LinkBinding
    from umr.retarget.pipeline import UMRRetargeter
    from scripts.umr_root_initialization_v6 import root_reference
    from diagnose_elf3_arm_motion import traces
    with np.load(stage1/'setup/bodies.npz',allow_pickle=True) as z:cloud=SurfacePointCloud.from_dict(z,'human_')
    with np.load(stage1/'setup/correspondence.npz',allow_pickle=True) as z:
        binding=LinkBinding.from_dict(z);segment=z['inherited_segment'].copy()
    class FullRetargeter(UMRRetargeter):
        def human_targets(self, frame):
            p,n=self.human.targets(frame)
            return p[sampled],n[sampled]

        def initialize_root(self, frame):
            self.robot.set_tpose();self.human.set_tpose()
            offset=self.robot.data.body('torso_link').xpos-self.human.data.xpos[9]
            self.human.set_frame(frame)
            trace=root_reference(self.human.data.xmat[0].reshape(3,3),prepared['metadata'])
            rotation=Rotation.from_euler('z',trace['yaw_rad'])
            q=self.robot.model.key_qpos[0].copy()
            q[:3]=self.human.data.xpos[9]+rotation.apply(offset)
            q[3:7]=rotation.as_quat(scalar_first=True)
            q,foot=foot_feasible_initial_pose(self.robot.model,q,cfg.retarget['floor_height'],cfg.retarget['floor_margin'])
            self.initialization={'heading':trace,'foot':foot}
            self.robot.set_qpos(q)
    keys=('n_selected','point_selection','tpose_offset','iterations','dt','damping','solver','trust_region',
          'trust_region_radius','floor_height','floor_band','floor_margin','contact_threshold','contact_weight',
          'posture_cost','self_collision')
    with patch('umr.retarget.pipeline.sole_sample_points',elf3_mesh_sole_points):
        solver=FullRetargeter(robot,human,human_body_ids=cloud.body_ids,human_local_pos=cloud.local_pos,
            human_local_normal=cloud.local_normal,robot_body_ids=binding.body_ids,robot_local_pos=binding.local_pos,
            robot_local_normal=binding.local_normal,segment=segment,segment_names=SEGMENTS,
            **{k:cfg.retarget[k] for k in keys})
    solver.initialize_root(0);warm=solver.solve_frame(0,iterations=30)
    qpos=np.empty((len(prepared['times']),38));failures=[];errors=[]
    start=time.monotonic()
    for frame in range(len(qpos)):
        row=solver.solve_frame(frame)
        qpos[frame]=row['qpos'];failures.append(int(row['failures']));errors.append(row['point_error'])
        if frame%250==0:print('[stage2]',frame,len(qpos),'failures',sum(failures),flush=True)
    output.mkdir()
    inputs=json.loads((stage1/'inputs.json').read_text())
    inputs.update(source=str(source),source_sha256=digest(source),source_metadata=prepared['metadata'])
    write_json(output/'inputs.json',inputs)
    np.savez_compressed(output/'motion.npz',qpos=qpos,times=prepared['times'],fps=50.,
        dof_names=np.array([robot.model.joint(j).name for j in range(1,robot.model.njnt)]),
        root_body='torso_link',quaternion_order='wxyz',source_surface_indices=sampled,
        solve_failures=np.asarray(failures),point_error=np.asarray(errors))
    audit=motion_audit(robot.model,qpos,50.,urdf)
    _,mesh=traces(robot.model,qpos)
    receipt={'schema':'bfm.elf3_full_stage2/1','full_duration_verified':True,'frames':len(qpos),
        'stage1':str(stage1),'stage1_receipt_sha256':digest(stage1/'receipt.json'),
        'warmup_failures':int(warm['failures']),'solve_failures':sum(failures),
        'point_error_mean_m':float(np.mean(errors)),'initialization':solver.initialization,
        'audit':audit,'visual_screen':mesh,'seconds':time.monotonic()-start,
        'motion_sha256':digest(output/'motion.npz'),'source_clock_unchanged':True,
        'training_approved':False,'physical_tracking_validated':False,'policy_inference':False}
    write_json(output/'receipt.json',receipt)
    return receipt


def result_status(raw, refined):
    if raw['solve_failures'] or raw['warmup_failures']:return 'quarantined_ik_failure'
    if refined is None:return 'raw_complete_refinement_pending'
    audit=refined['audit']
    okay=(refined['optimizer_success'] and audit['joint_limit_violation_frames']==0
        and audit['joint_speed_over_urdf_limit_intervals']==0 and audit['foot_visual_mesh_min_z_m']>=0.
        and audit['original_collision_self_penetration_max_m']<=1e-6
        and refined['after_visual_screen']['visual_hull_max_penetration_m']<=1e-6
        and refined['tip_displacement_p95_m']<=.02 and refined['tip_displacement_max_m']<=.05
        and refined['raw_human_surface_error_mean_m']<=refined['input_raw_human_surface_error_mean_m']+.005)
    return 'kinematic_review_required' if okay else 'quarantined_quality'


def preparation_command(row, config, source):
    return [config['prepare_python'],str(ROOT/'scripts/umr_heading_source_v6.py'),'--source',row['source'],
            '--body-model',config['body_model'],'--output',str(source)]


def run(args):
    from run_elf3_library import verify_frozen, child
    plan=json.loads(args.plan.read_text());verify_frozen(plan)
    row=plan['jobs'][args.index];config=plan['config'];output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    if digest(row['source'])!=row['source_sha256']:raise ValueError('Raw source changed')
    verify_assets(Path(config['assets']))
    source=output/'prepared.npz'
    command=preparation_command(row,config,source)
    code=child(command,output/'prepare.log',1200,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    if code:raise RuntimeError(f'Full source preparation failed: {code}')
    prepared=load_replay_source(source,require_full=True)
    from scripts.umr_heading_source_v6 import validate_values
    validate_values({k:v for k,v in prepared.items() if k!='metadata'})
    if (prepared['metadata']['source_sha256']!=row['source_sha256']
            or len(prepared['times'])!=row['target_50hz_frames']):raise ValueError('Source clock/identity mismatch')
    cfg,robot,human,cloud,sampled=stage1_context(prepared,config['robot_xml'])
    stage1=obtain_stage1(Path(config['stage1_cache']),prepared,source,cfg,robot,human,cloud,sampled,plan['protected'])
    raw=solve_full(prepared,source,stage1,output/'raw',cfg,robot,human,sampled,Path(config['urdf']))
    refined=None;reason=None
    if not raw['solve_failures'] and not raw['warmup_failures'] and 4<=raw['frames']<=1501:
        command=[sys.executable,str(ROOT/'scripts/refine_elf3_whole_body.py'),'--source-run',str(output/'raw'),
            '--stage1',str(stage1),'--urdf',config['urdf'],'--output',str(output/'whole_body'),
            '--max-iterations','450','--max-seconds','600']
        code=child(command,output/'whole_body.log',900,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        if code==0:refined=json.loads((output/'whole_body/receipt.json').read_text())
        else:reason=f'whole_body_exit_{code}; raw preserved, not accepted'
    elif raw['frames']>1501:reason='full Stage II complete; long continuous refinement not yet supported'
    verify_frozen(plan)
    verify_assets(Path(config['assets']))
    artifacts={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    result={'schema':'bfm.elf3_library_job/1','source_sha256':row['source_sha256'],'origin_id':row['origin_id'],
        'split':row['split'],'frames':raw['frames'],'stage2_complete':True,'refinement_complete':refined is not None,
        'status':result_status(raw,refined),'reason':reason,'artifacts':artifacts,
        'stage1':str(stage1),'training_approved':False,'policy_training_started':False}
    write_json(output/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
