#!/usr/bin/env python3
"""Bounded whole-body spline experiment with restartable coefficient checkpoints.

Inputs remain untouched; numerical completion never grants training approval.
Checkpoint restart preserves coefficients/recipe, NOT L-BFGS history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np

from diagnose_elf3_arm_motion import motion_metrics, traces, visual_arm_core_pairs
from elf3_source_reuse import load_replay_source, require_same_canonical
from elf3_umr_asset import digest, write_json
from elf3_whole_body_model import WholeBodyObjective
from refine_elf3_arm_spline import spline_basis
from refine_elf3_umr_trajectory import BoundSurface
from run_elf3_umr_trial import UMR, elf3_mesh_sole_points, motion_audit
from umr_smplx_source import SEGMENTS, SmplxSurfaceHuman, model_geometry_fingerprint, verify_umr_checkout


def joint_temporal_metrics(model, qpos):
    result = {}
    for j in range(1, model.njnt):
        values = qpos[:, model.jnt_qposadr[j]]
        delta = np.diff(values)
        velocity = delta*50.
        acceleration = np.diff(velocity)*50.
        i = int(np.abs(delta).argmax())
        result[model.joint(j).name] = {"step_max_deg": float(np.rad2deg(np.abs(delta).max())),
            "speed_max_rad_s": float(np.abs(velocity).max()), "acceleration_max_rad_s2": float(np.abs(acceleration).max()),
            "acceleration_rms_rad_s2": float(np.sqrt(np.mean(acceleration**2))), "peak_step_interval": [i,i+1]}
    return result


def input_ik_valid(receipt):
    if receipt.get('schema') == 'bfm.elf3_full_stage2/1':
        return (receipt.get('full_duration_verified') is True and receipt.get('solve_failures') == 0
                and receipt.get('warmup_failures') == 0)
    return receipt.get('kinematic_checks',{}).get('input_umr_no_solver_failures') is True


def run(args):
    import mujoco
    import mink
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    from umr.bodies.robot import RobotBody, RobotSpec
    from umr.bodies.surface import SurfacePointCloud, transport_points
    from umr.retarget.binding import LinkBinding
    from umr.retarget.pipeline import select_correspondence_points

    verify_umr_checkout(UMR)
    folder, stage1, output = (p.resolve() for p in (args.source_run,args.stage1,args.output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    inputs=json.loads((folder/'inputs.json').read_text())
    parent=json.loads((folder/'receipt.json').read_text())
    seed_inputs=json.loads((stage1/'inputs.json').read_text())
    if parent['schema'] not in ('bfm.elf3_trajectory_refinement/1','bfm.elf3_arm_spline_refinement/1','bfm.elf3_full_stage2/1'):
        raise ValueError("Use a verified non-frozen refined trajectory, not failed raw IK")
    original_parent = parent
    if parent['schema']=='bfm.elf3_arm_spline_refinement/1':
        origin=Path(parent['source_run'])
        if digest(origin/'motion.npz') != parent['protected_inputs'][str(origin/'motion.npz')]:
            raise ValueError("Arm candidate original changed")
        original_parent=json.loads((origin/'receipt.json').read_text())
    if not input_ik_valid(original_parent):
        raise ValueError("Failed input IK must be fixed before trajectory optimization")
    if (digest(folder/'motion.npz') != parent['motion_sha256']
        or digest(inputs['source']) != inputs['source_sha256']
        or digest(seed_inputs['source']) != seed_inputs['source_sha256']
        or digest(inputs['robot_xml']) != inputs['robot_xml_sha256']
        or inputs['robot_geometry'] != seed_inputs['robot_geometry']):
        raise ValueError("Source, robot or Stage I identity changed")
    protected={str(p):digest(p) for p in [folder/'inputs.json',folder/'receipt.json',folder/'motion.npz',
        stage1/'inputs.json',stage1/'receipt.json',stage1/'setup/bodies.npz',stage1/'setup/correspondence.npz',
        Path(inputs['source']),Path(seed_inputs['source']),Path(inputs['robot_xml']),args.urdf.resolve(),
        *[Path(__file__).with_name(name) for name in ('refine_elf3_whole_body.py','elf3_whole_body_model.py',
         'refine_elf3_arm_spline.py','refine_elf3_umr_trajectory.py','diagnose_elf3_arm_motion.py',
         'elf3_source_reuse.py','elf3_umr_asset.py','run_elf3_umr_trial.py','umr_smplx_source.py','elf3_mesh_distance.py')]]}
    robot=RobotBody(inputs['robot_xml'],RobotSpec.from_config(inputs['config']['robot']));model=robot.model
    if model_geometry_fingerprint(model) != inputs['robot_geometry']:
        raise ValueError("Mesh identity changed")
    source=load_replay_source(inputs['source'],require_full=True)
    require_same_canonical(load_replay_source(seed_inputs['source'],require_full=True),source)
    human=SmplxSurfaceHuman(source,robot.height())
    with np.load(folder/'motion.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    qpos=arrays['qpos'];samples=arrays['source_surface_indices']
    if (qpos.ndim != 2 or qpos.shape[1] != 38 or not 4<=len(qpos)<=1501
        or arrays['dof_names'].tolist() != [model.joint(j).name for j in range(1,model.njnt)]
        or str(arrays['root_body'].item()) != 'torso_link' or str(arrays['quaternion_order'].item()) != 'wxyz'
        or float(arrays['fps']) != 50. or not np.array_equal(arrays['times'],source['times'])):
        raise ValueError("Complete 31-DoF source clock/conventions required")
    with np.load(stage1/'setup/bodies.npz',allow_pickle=True) as z:cloud=SurfacePointCloud.from_dict(z,'human_')
    with np.load(stage1/'setup/correspondence.npz',allow_pickle=True) as z:
        binding,segment=LinkBinding.from_dict(z),z['inherited_segment'].copy()
    if (len(samples)!=len(cloud.points) or samples.dtype.kind not in 'iu' or np.any(samples<0)
        or np.any(samples>=len(source['canonical_points'])) or len(np.unique(samples)) != len(samples)):
        raise ValueError("Correspondence sampling identity differs")
    canonical,_=transport_points(human.data,cloud.body_ids,cloud.local_pos,cloud.local_normal)
    selected=select_correspondence_points(segment,SEGMENTS,inputs['config']['retarget']['n_selected'],
        points=canonical,method=inputs['config']['retarget']['point_selection'])
    surface=BoundSurface(model,binding.body_ids[selected],binding.local_pos[selected])
    geoms=mink.get_subtree_geom_ids(model,1)
    original_pairs=list(mink.CollisionAvoidanceLimit(model,[(geoms,geoms)]).geom_id_pairs)
    basis,knots=spline_basis(arrays['times'],args.knot_seconds)
    objective=WholeBodyObjective(model,qpos,surface,original_pairs,visual_arm_core_pairs(model),basis,args.clearance_m)
    coefficients=np.linalg.lstsq(basis,objective.chart.original,rcond=None)[0]
    limits=model.jnt_range[1:]
    coefficients[:,6:]=np.maximum(np.minimum(coefficients[:,6:],limits[:,1]),limits[:,0])
    recipe={'schema':'bfm.elf3_whole_body_recipe/1','protected_inputs':protected,
        'source_run':str(folder),'knot_seconds':args.knot_seconds,'visual_soft_clearance_m':args.clearance_m,
        'objective':'full_body_so3_chart_surface_feet_wrists_temporal_v1'}
    if args.initial_checkpoint is not None:
        metadata=json.loads(args.initial_checkpoint.with_suffix('.json').read_text())
        if metadata['recipe'] != recipe or digest(args.initial_checkpoint) != metadata['sha256']:
            raise ValueError("Checkpoint code/source/recipe changed")
        with np.load(args.initial_checkpoint) as z:
            if not np.array_equal(z['knots'],knots):raise ValueError("Checkpoint clock changed")
            coefficients=z['coefficients'].copy()
    if coefficients.shape != (basis.shape[1],37) or not np.isfinite(coefficients).all():
        raise ValueError("Invalid coefficients")
    output.mkdir(parents=True)
    write_json(output/'recipe.json',recipe)
    stopped=[False]
    previous_handlers={s:signal.signal(s,lambda signum,frame:stopped.__setitem__(0,True)) for s in (signal.SIGTERM,signal.SIGINT)}
    history=[];start=time.monotonic()

    def checkpoint(x):
        path=output/f'checkpoint_{len(history):05d}.npz'
        if not path.exists():
            np.savez_compressed(path,coefficients=x.reshape(coefficients.shape),knots=knots)
            write_json(path.with_suffix('.json'),{'sha256':digest(path),'recipe':recipe,
                'iteration':len(history),'optimizer_history_restored':False})

    def callback(x):
        history.append({'iteration':len(history)+1,**objective.latest})
        if len(history)%25==0:
            checkpoint(x);print(json.dumps(history[-1]),flush=True)
        if stopped[0] or time.monotonic()-start>=args.max_seconds:
            checkpoint(x)
            raise StopIteration

    bounds=[(None,None)]*6+list(map(tuple,limits))
    try:
        initial=objective(coefficients.ravel())[0]
        result=minimize(objective,coefficients.ravel(),jac=True,method='L-BFGS-B',
            bounds=bounds*len(coefficients),callback=callback,
            options={'maxiter':args.max_iterations,'ftol':1e-10,'gtol':1e-7,'maxls':30,'maxcor':12})
        checkpoint(result.x)
        optimized=objective.chart.qpos(basis@result.x.reshape(coefficients.shape))
        audit=motion_audit(model,optimized,50.,args.urdf)
        before,before_mesh=traces(model,qpos);after,after_mesh=traces(model,optimized)
        arm_metrics={side:{'before':motion_metrics(before['tool_positions'][:,i],before['wrist_rotations'][:,i]),
                           'after':motion_metrics(after['tool_positions'][:,i],after['wrist_rotations'][:,i])}
                     for i,side in enumerate(('left','right'))}
        raw_errors=[];old_errors=[];surface_delta=[];feet_delta=[]
        for i,q in enumerate(optimized):
            objective.data.qpos[:]=q;mujoco.mj_forward(model,objective.data)
            points,_=surface.evaluate(objective.data,False);raw,_=human.targets(i);target=raw[samples][selected]
            raw_errors.append(np.linalg.norm(points-target,axis=1).mean())
            old_errors.append(np.linalg.norm(objective.target[i]-target,axis=1).mean())
            surface_delta.extend(np.linalg.norm(points-objective.target[i],axis=1).tolist())
            foot,_=objective.feet.evaluate(objective.data,False)
            feet_delta.extend(np.linalg.norm(foot-objective.foot_target[i],axis=1).tolist())
        tips_delta=np.linalg.norm(after['tool_positions']-before['tool_positions'],axis=2)
        for key in ('point_error','solve_failures'):arrays.pop(key,None)
        arrays['qpos']=optimized
        np.savez_compressed(output/'motion.npz',**arrays)
        np.savez_compressed(output/'spline.npz',knots=knots,coefficients=result.x.reshape(coefficients.shape),
            root_reference_wxyz=objective.chart.reference.as_quat(scalar_first=True),degree=3)
        for path,checksum in protected.items():
            if digest(path)!=checksum:raise ValueError("Protected input/code changed")
        if model_geometry_fingerprint(mujoco.MjModel.from_xml_path(inputs['robot_xml'])) != inputs['robot_geometry']:
            raise ValueError("On-disk mesh changed during optimization")
        write_json(output/'inputs.json',inputs);write_json(output/'optimizer_history.json',history)
        receipt={'schema':'bfm.elf3_whole_body_spline/1','source_run':str(folder),'protected_inputs':protected,
            'optimizer_success':bool(result.success),'optimizer_message':str(result.message),
            'iterations':int(result.nit),'seconds':time.monotonic()-start,'initial_loss':initial,'final_loss':float(result.fun),
            'root_and_all_31_joints_optimized':True,'knot_seconds':args.knot_seconds,'visual_soft_clearance_m':args.clearance_m,
            'arm_metrics':arm_metrics,'before_visual_screen':before_mesh,'after_visual_screen':after_mesh,'audit':audit,
            'joint_metrics':{'before':joint_temporal_metrics(model,qpos),'after':joint_temporal_metrics(model,optimized)},
            'root_metrics':{label:motion_metrics(q[:,:3],Rotation.from_quat(q[:,3:7],scalar_first=True).as_matrix())
                           for label,q in (('before',qpos),('after',optimized))},
            'surface_change_p95_m':float(np.percentile(surface_delta,95)),
            'tip_displacement_p95_m':float(np.percentile(tips_delta,95)),'tip_displacement_max_m':float(tips_delta.max()),
            'feet_displacement_p95_m':float(np.percentile(feet_delta,95)),'feet_displacement_max_m':float(np.max(feet_delta)),
            'raw_human_surface_error_mean_m':float(np.mean(raw_errors)),
            'input_raw_human_surface_error_mean_m':float(np.mean(old_errors)),
            'motion_sha256':digest(output/'motion.npz'),'source_clock_unchanged':True,'training_approved':False,
            'physical_tracking_validated':False,'policy_inference':False,'arm_review_complete':False,
            'new_neural_training':False,'promoted_to_training':False,
            'checkpoint_restart':'coefficients_and_recipe_only_not_LBFGS_history'}
        write_json(output/'receipt.json',receipt)
        print(json.dumps(receipt,indent=2),flush=True)
    finally:
        for s,handler in previous_handlers.items():signal.signal(s,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True)
    p.add_argument('--urdf',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--knot-seconds',type=float,default=.12);p.add_argument('--clearance-m',type=float,default=.001)
    p.add_argument('--max-iterations',type=int,default=800);p.add_argument('--max-seconds',type=int,default=1200)
    p.add_argument('--initial-checkpoint',type=Path)
    args=p.parse_args()
    if not 10<=args.max_iterations<=1500 or not 30<=args.max_seconds<=3600:raise ValueError('Bounded experiment required')
    sys.path.insert(0,str(UMR));run(args)
