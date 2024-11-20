import sys

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

import cv2
from tqdm import tqdm
from skimage import io
from datetime import datetime

import matplotlib.pyplot as plt

from core import datasets
from core.utils import flow_viz
from core.utils import frame_utils

from core.flowdiffuser import FlowDiffuser
from core.utils.utils import InputPadder, forward_interpolate


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            # if sequence != sequence_prev:
            #     flow_prev = None
            
            if (sequence != sequence_prev) or (dstype == 'final' and sequence in ['market_4', ]) or dstype == 'clean':
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        result_dir = os.path.join(args.result_dir, dstype)
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype,
                                root= os.path.join(args.data_dir, "Sintel")
                                )
        epe_list = []
        
        for val_id in tqdm(range(len(val_dataset))):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            
            # save flow
            if val_id % 100 == 0:
                index = 0
                flow_est_b = flow.numpy().transpose((1, 2, 0))
                
                os.makedirs(result_dir, exist_ok=True)
                flo_path = os.path.join(result_dir, f"{val_id:05d}")
                #frame_utils.writeFlow( flo_path + ".flo",  flow_est_b)
                flo_clr = flow_viz.flow_to_image(flow_est_b)
                flo_clr = frame_utils.add_text_to_image(
                    text_str= f'epe: {epe_list[-1].mean():.2f}', 
                    posi= (20, 30),#(x,y) 
                    img = flo_clr,
                    fontScale= 0.6,
                    fontColor= (0,0,0)
                    )
                io.imsave(flo_path + "-flow-clr.png", flo_clr)
            
            #print (f"processing {dstype} data {val_id + 1}/{len(val_dataset)}")
            #if val_id > 2:
            #    break 
            

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)
        results.update({
            f'{dstype}/epe': epe, 
            f'{dstype}/1px': px1,
            f'{dstype}/3px': px3,
            f'{dstype}/5px': px5,
            f'{dstype}/imgN': len(epe_list),
            })

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training',
                                root= os.path.join(args.data_dir, "KITTI_2015")
                                )

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu() #[2,H,W]
        #print (f"??? flow shape = {flow.shape}")

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
            
        # save flow
        flow_est_b = flow.numpy().transpose((1, 2, 0))
        os.makedirs(args.result_dir, exist_ok=True)
        flo_path = os.path.join(args.result_dir, f"{val_id:05d}")
        #frame_utils.writeFlow( flo_path + ".flo",  flow_est_b)
        flo_clr = flow_viz.flow_to_image(flow_est_b)
        flo_clr = frame_utils.add_text_to_image(
            text_str= f'epe: {epe_list[-1]:.2f}', 
            posi= (20, 30),#(x,y) 
            img = flo_clr,
            fontScale= 0.6,
            fontColor= (0,0,0)
            )
        io.imsave(flo_path + "-flow-clr.png", flo_clr)
        #print (f"processing data {val_id + 1}/{len(val_dataset)}")
        
        #if val_id > 2:
        #    break

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1, 'imgN': len(epe_list)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    # added by CCJ;
    parser.add_argument('--data_dir', type = str, default= "datasets/", help="input data dir")
    parser.add_argument('--result_dir', type = str, default= "./results", help="result dir")
    parser.add_argument('--machine_name', type = str, help="result dir")
    parser.add_argument('--eval_gpu_id', type = str, default='0', help="result dir")
    parser.add_argument('--model_name', type = str, default='fdm', help="model name")
    
    args = parser.parse_args()

    model = torch.nn.DataParallel(FlowDiffuser(args))
    model.load_state_dict(torch.load(args.model, weights_only=False))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    results = {}
    val_dataset = args.dataset
    
    if args.result_dir.find("results_nfs/") >= 0:
        pos = args.result_dir.find("results_nfs/")
        general_csv_root = args.result_dir[:pos+len("results_nfs/")]
    elif args.result_dir.find("results/") >= 0:
        pos = args.result_dir.find("results/")
        general_csv_root = args.result_dir[:pos+len("results/")]
    else:
        raise NotImplementedError

    with torch.no_grad():
        if args.dataset == 'chairs':
            results.update(validate_chairs(model.module))

        elif args.dataset == 'sintel':
            results.update(validate_sintel(model.module)) 
            timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            os.makedirs(args.result_dir, exist_ok=True)
            csv_file = os.path.join(args.result_dir, f"{val_dataset}-err.csv")

            messg = timeStamp + ",model={},resultDir,{},dataset,{},data_root,{},".format(
                    args.model_name, args.result_dir, val_dataset, args.data_dir) + \
                    (",epe(clean),{},1px(clean),{},3px(clean),{},5px(clean),{}").format(
                        results['clean/epe'], 
                        results['clean/1px'], 
                        results['clean/3px'], 
                        results['clean/5px'], 
                    ) + \
                    (",epe(final),{},1px(final),{},3px(final),{},5px(final),{}").format(
                        results['final/epe'], 
                        results['final/1px'], 
                        results['final/3px'], 
                        results['final/5px'], 
                    ) + \
                    (",,Frames number {} (clean) and {} (final) for evalution".format(
                        results['clean/imgN'], results['final/imgN']
                        )
                    ) + "\n"
            
            print (messg)
            with open( csv_file, 'w') as fwrite:
                fwrite.write(messg + "\n")
            dst_csv_file = os.path.join(general_csv_root, 'sintel_eval_err.csv') 
            print (dst_csv_file)
            os.system(f'cat {csv_file} >> {dst_csv_file}')

        elif args.dataset == 'kitti':
            results.update(validate_kitti(model.module))
            
            timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            csv_file = os.path.join(args.result_dir, f"{val_dataset}-err.csv")
            messg = timeStamp + f",model={args.model_name},resultDir,{args.result_dir}" + \
                    f",dataset,{val_dataset},data_root,{args.data_dir}," + \
                    f",epe,{results['kitti-epe']},F1,{results['kitti-f1']}" + \
                    f",,Frames number {results['imgN']} for evalution\n"

            print (messg)
            with open( csv_file, 'w') as fwrite:
                fwrite.write(messg)
            dst_csv_file = os.path.join(general_csv_root, 'kt15_eval_err.csv') 
            print (dst_csv_file)
            os.system(f'cat {csv_file} >> {dst_csv_file}')  


