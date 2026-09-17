import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_elf3_library import ordered_jobs, counts, verified_result, atomic_json, child
from elf3_library_worker import result_status, preparation_command
from refine_elf3_whole_body import input_ik_valid
from elf3_umr_asset import digest


class LibraryTests(unittest.TestCase):
    def job(self,i,frames=100):
        return {'source_sha256':f'{i:064x}','split':'train','output_robot':'elf3_dof31',
                'target_50hz_frames':frames,'shape_betas_sha256':'shape','origin_id':f'actor/motion{i}'}

    def test_full_plan_keeps_long_sources_and_splits(self):
        jobs=[self.job(1),self.job(2,13182)];jobs[1]['split']='validation'
        result=ordered_jobs({'schema':'bfm.elf3_batch_plan/1','jobs':jobs,'unique_jobs':2})
        self.assertEqual(result,jobs)

    def test_source_preparation_uses_existing_separate_environment(self):
        command=preparation_command({'source':'source.npz'},
            {'prepare_python':'/venv-smplx/bin/python','body_model':'model.npz'},Path('out.npz'))
        self.assertEqual(command[0],'/venv-smplx/bin/python')
        self.assertEqual(command[2],'--source')
        self.assertNotIn('prepare',command)

    def test_duplicate_sources_rejected(self):
        with self.assertRaises(ValueError):
            ordered_jobs({'schema':'bfm.elf3_batch_plan/1','jobs':[self.job(1),self.job(1)],'unique_jobs':2})

    def test_counts_do_not_call_failures_or_refinement_pending_accepted(self):
        jobs=[self.job(i) for i in range(4)]
        finished={0:{'status':'execution_failed','stage2_complete':False},
                  1:{'status':'raw_complete_refinement_pending','stage2_complete':True}}
        r=counts(jobs,finished,{2:None})
        self.assertEqual(r['pending_sources'],1);self.assertEqual(r['stage2_complete'],1)
        self.assertEqual(r['training_approved'],0);self.assertFalse(r['all_library_retargeted'])

    def test_parent_full_clock_and_all_ik_success_required(self):
        receipt={'schema':'bfm.elf3_full_stage2/1','full_duration_verified':True,'solve_failures':0,'warmup_failures':0}
        self.assertTrue(input_ik_valid(receipt))
        for key,value in [('solve_failures',1),('warmup_failures',1),('full_duration_verified',False)]:
            self.assertFalse(input_ik_valid({**receipt,key:value}))

    def test_quality_failure_is_not_training_acceptance(self):
        raw={'solve_failures':0,'warmup_failures':0}
        self.assertEqual(result_status(raw,None),'raw_complete_refinement_pending')
        self.assertEqual(result_status({**raw,'solve_failures':1},None),'quarantined_ik_failure')
        good={'optimizer_success':True,'audit':{'joint_limit_violation_frames':0,
              'joint_speed_over_urdf_limit_intervals':0,'foot_visual_mesh_min_z_m':.001,
              'original_collision_self_penetration_max_m':0.},'after_visual_screen':{'visual_hull_max_penetration_m':0.},
              'tip_displacement_p95_m':.01,'tip_displacement_max_m':.02,
              'raw_human_surface_error_mean_m':.06,'input_raw_human_surface_error_mean_m':.06}
        self.assertEqual(result_status(raw,good),'kinematic_review_required')
        bad=copy.deepcopy(good);bad['audit']['foot_visual_mesh_min_z_m']=-.001
        self.assertEqual(result_status(raw,bad),'quarantined_quality')
        bad=copy.deepcopy(good);bad['optimizer_success']=False
        self.assertEqual(result_status(raw,bad),'quarantined_quality')

    def test_result_hash_and_original_identity_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);artifact=p/'output.npz';artifact.write_bytes(b'original');row=self.job(1)
            receipt={'source_sha256':row['source_sha256'],'split':'train','frames':100,'training_approved':False,
                     'stage2_complete':False,'artifacts':{'output.npz':digest(artifact)}}
            atomic_json(p/'result.json',receipt);self.assertEqual(verified_result(p/'result.json',row),receipt)
            artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError):verified_result(p/'result.json',row)

    def test_child_timeout_reaped_and_atomic_status_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            self.assertEqual(child([sys.executable,'-c','import time;time.sleep(30)'],p/'run.log',.05),124)
            atomic_json(p/'status.json',{'count':1});atomic_json(p/'status.json',{'count':2})
            self.assertEqual(json.loads((p/'status.json').read_text()),{'count':2})


if __name__=='__main__':unittest.main()
