import time
import logging
import argparse
import importlib
import torch
import torch.distributed
import torch.backends.cudnn as cudnn
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

from mmdet.datasets import replace_ImageToTensor
from loaders.builder import build_iter_dataloader

def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--num_warmup', default=40)
    parser.add_argument('--samples', default=400)
    parser.add_argument('--log-interval', default=50, help='interval of logging')
    parser.add_argument('--override', nargs='+', action=DictAction)
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # you need GPUs
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    logging.info('Using GPU: %s' % torch.cuda.get_device_name(0))
    torch.cuda.set_device(0)

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)

    distributed = False
    # in case the test dataset is concatenated
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    if isinstance(cfgs.data.test, dict):
        cfgs.data.test.test_mode = True
        if cfgs.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfgs.data.test.pipeline = replace_ImageToTensor(
                cfgs.data.test.pipeline)
    elif isinstance(cfgs.data.test, list):
        for ds_cfg in cfgs.data.test:
            ds_cfg.test_mode = True
        if cfgs.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfgs.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfgs.data.get('test_dataloader', {})
    }

    # build the dataloader

    test_dataset = build_dataset(cfgs.data.test)
    
    test_loader = build_iter_dataloader(test_dataset, **test_loader_cfg)

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model, test_cfg=cfgs.get('test_cfg'))
    model.cuda()

    assert torch.cuda.device_count() == 1
    model = MMDataParallel(model, [0])

    logging.info('Loading checkpoint from %s' % args.checkpoint)
    fp16_cfg = cfgs.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', revise_keys=[(r'^module\.', ''), (r'^teacher\.', '')])
    
    model.eval()

    print('Timing w/o data loading:')
    pure_inf_time = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= args.num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % args.log_interval == 0:
                    fps = (i + 1 - args.num_warmup) / pure_inf_time
                    print(f'Done sample [{i + 1:<3}/ {args.samples}], '
                        f'fps: {fps:.1f} sample / s')

            if (i + 1) == args.samples:
                break


if __name__ == '__main__':
    main()
