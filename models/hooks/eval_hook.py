# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


import bisect
import os.path as osp
from .utils import is_parallel
import mmcv
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core.evaluation.eval_hooks import DistEvalHook


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class CustomDistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None,  **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if runner.model.module.__class__.__name__ == "TemporalOcc"\
                     and runner.model.module.pts_bbox_head.use_history_flag == True:
            if is_parallel(runner.model):
                if  runner.model.module.pts_bbox_head.history_occ is not None:
                    history_occ = runner.model.module.pts_bbox_head.history_occ.clone()
                    history_seq_ids = runner.model.module.pts_bbox_head.history_seq_ids.clone()
                    history_forward_augs = runner.model.module.pts_bbox_head.history_forward_augs.clone()
                    history_sweep_time = runner.model.module.pts_bbox_head.history_sweep_time.clone()
                else:
                    history_occ = None
                    
                runner.model.module.pts_bbox_head.history_occ=None
                runner.ema_model.ema_model.module.pts_bbox_head.history_occ=None
            else:
                runner.ema_model.ema_model.pts_bbox_head.history_occ=None
                runner.model.pts_bbox_head.history_occ = None
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmdet.apis import multi_gpu_test

        # Changed results to self.results so that MMDetWandbHook can access
        results = multi_gpu_test(
            runner.ema_model.ema_model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        self.latest_results = results
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            # the key_score may be `None` so it needs to skip
            # the action to save the best checkpoint
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
        
        if runner.model.module.__class__.__name__ == "TemporalOcc"\
                     and runner.model.module.pts_bbox_head.use_history_flag == True:
            if is_parallel(runner.model):
                if history_occ is not None:
                    runner.model.module.pts_bbox_head.history_occ = history_occ.clone()
                    runner.model.module.pts_bbox_head.history_seq_ids = history_seq_ids.clone()
                    runner.model.module.pts_bbox_head.history_forward_augs = history_forward_augs.clone()
                    runner.model.module.pts_bbox_head.history_sweep_time = history_sweep_time.clone()
                else:
                    runner.model.module.pts_bbox_head.history_occ = None
                runner.ema_model.ema_model.module.pts_bbox_head.history_occ = None
            else:
                runner.model.pts_bbox_head.history_occ = None
                runner.ema_model.ema_model.pts_bbox_head.history_occ = None

