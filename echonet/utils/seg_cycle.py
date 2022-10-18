"""Functions for training and running segmentation."""

import math
import os
import time
import shutil
import datetime
import pandas as pd

import click
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import skimage.draw
from PIL import Image
import torch
import torchvision
import tqdm

import echonet


@click.command("seg_cycle")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.segmentation.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.segmentation.__dict__[name]))),
    default="deeplabv3_resnet50")
@click.option("--pretrained/--random", default=False)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--save_video", type=str, default=None)
@click.option("--num_epochs", type=int, default=25) 
@click.option("--lr", type=float, default=1e-5)
@click.option("--weight_decay", type=float, default=0)
@click.option("--lr_step_period", type=int, default=None)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
@click.option("--reduced_set/--full_set", default=True)
@click.option("--rd_label", type=int, default=920)  
@click.option("--rd_unlabel", type=int, default=6440) 
@click.option("--ssl_edesonly/--ssl_rndfrm", default=True)
@click.option("--run_inference", type=str, default=None)
@click.option("--chunk_size", type=int, default=3)
@click.option("--cyc_off", type=int, default=2)
@click.option("--target_region", type=int, default=15)
@click.option("--temperature", type=int, default=10)
@click.option("--val_chunk", type=int, default=40)
@click.option("--loss_cyc_w", type=float, default=1)
@click.option("--css_strtup", type=int, default=0)

def run(
    data_dir=None,
    output=None,
    model_name="deeplabv3_resnet50",
    pretrained=False,
    weights=None,
    run_test=False,
    save_video=None,
    num_epochs=25,
    lr=1e-5,
    weight_decay=1e-5,
    lr_step_period=None,
    num_train_patients=None,
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
    reduced_set = True,
    rd_label = 920,
    rd_unlabel = 6440,
    ssl_edesonly = True,
    run_inference = None,
    chunk_size = 3,
    cyc_off = 2,
    target_region = 15,
    temperature = 10,
    val_chunk = 40,
    loss_cyc_w = 1,
    css_strtup = 0
):

    if reduced_set:
        if not os.path.isfile(os.path.join(echonet.config.DATA_DIR, "FileList_ssl_{}_{}.csv".format(rd_label, rd_unlabel))):
            print("Generating new file list for ssl dataset")
            np.random.seed(0)
            
            data = pd.read_csv(os.path.join(echonet.config.DATA_DIR, "FileList.csv"))
            data["Split"].map(lambda x: x.upper())

            file_name_list = np.array(data[data['Split']== 'TRAIN']['FileName'])
            np.random.shuffle(file_name_list)

            label_list = file_name_list[:rd_label]
            unlabel_list = file_name_list[rd_label:rd_label + rd_unlabel]

            data['SSL_SPLIT'] = "EXCLUDE"
            data.loc[data['FileName'].isin(label_list), 'SSL_SPLIT'] = "LABELED"
            data.loc[data['FileName'].isin(unlabel_list), 'SSL_SPLIT'] = "UNLABELED"

            data.to_csv(os.path.join(echonet.config.DATA_DIR, "FileList_ssl_{}_{}.csv".format(rd_label, rd_unlabel)),index = False)


    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    def worker_init_fn(worker_id):                            
        # print("worker id is", torch.utils.data.get_worker_info().id)
        # https://discuss.pytorch.org/t/in-what-order-do-dataloader-workers-do-their-job/88288/2
        np.random.seed(np.random.get_state()[1][0] + worker_id)


    # Set default output directory
    if output is None:
        output = os.path.join("output", "segmentation", "{}_{}".format(model_name, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    bkup_tmstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.path.isdir(os.path.join(output, "echonet_{}".format(bkup_tmstmp))):
        shutil.rmtree(os.path.join(output, "echonet_{}".format(bkup_tmstmp)))
    shutil.copytree("echonet", os.path.join(output, "echonet_{}".format(bkup_tmstmp)))


    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "gpu":
        device = torch.device("cuda")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        assert 1==2, "wrong parameter for device"



    #### Setup model
    model_0 = echonet.segmentation.segmentation.deeplabv3_resnet50_CSS(pretrained=pretrained, aux_loss=False)
    model_0.classifier[-1] = torch.nn.Conv2d(model_0.classifier[-1].in_channels, 1, kernel_size=model_0.classifier[-1].kernel_size)  # change number of outputs to 1
    model_0 = torch.nn.DataParallel(model_0)
    model_0.to(device) 

    if weights:
        checkpoint = torch.load(weights)
        model_0.load_state_dict(checkpoint['state_dict_0'], strict = False)
    
    # Set up optimizer
    optim_0 = torch.optim.SGD(model_0.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler_0 = torch.optim.lr_scheduler.StepLR(optim_0, lr_step_period)


    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    tasks_eval = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs_eval = {"target_type": tasks_eval,
              "mean": mean,
              "std": std
              }

    tasks_seg = ["LargeFrame", "SmallFrame", "LargeTrace", "SmallTrace"]
    kwargs_seg = {"target_type": tasks_seg,
              "mean": mean,
              "std": std
              }

    kwargs = {"target_type": ["EF", "CYCLE"],
              "mean": mean,
              "std": std,
              "length": 40,
              "period": 3,
              }


    dataset = {}
    dataset_trainsub = {}
    dataset_valsub = {}
    if reduced_set:
        dataset_trainsub['lb_seg'] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs_seg, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 1, ssl_edesonly = True)
        dataset_trainsub['lb_cyc'] = echonet.datasets.Echo_CSS(root=data_dir, split="train", **kwargs, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 1)
        dataset_trainsub['unlb_cyc'] = echonet.datasets.Echo_CSS(root=data_dir, split="train", **kwargs, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 2)
    else:
        assert not ssl_edesonly, "Check parameters, trying to conduct ssl with full datasest with EDES only"
        dataset_trainsub['lb_seg'] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs_seg, ssl_postfix="", ssl_type = 0, ssl_edesonly = True)
        dataset_trainsub['lb_cyc'] = echonet.datasets.Echo_CSS(root=data_dir, split="train", **kwargs, ssl_postfix="", ssl_type = 0)
        dataset_trainsub['unlb_cyc'] = echonet.datasets.Echo_CSS(root=data_dir, split="train", **kwargs, ssl_postfix="", ssl_type = 0)
    dataset['train'] = dataset_trainsub
    

    if reduced_set:
        dataset_valsub["lb_seg"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs_seg, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel))
        dataset_valsub['lb_cyc'] = echonet.datasets.Echo_CSS(root=data_dir, split="val", **kwargs, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel))
        dataset_valsub['unlb_cyc'] = echonet.datasets.Echo_CSS(root=data_dir, split="val", **kwargs, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel))
    else:
        assert 1 == 2, "only run with reduced set for now "
        dataset["val"] = echonet.datasets.Echo_CSS(root=data_dir, split="val", **kwargs, ssl_postfix="")
    dataset['val'] = dataset_valsub


    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:

        f.write("Run timestamp: {}\n".format(bkup_tmstmp))

        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            print("checkpoint.keys", checkpoint.keys())
            model_0.load_state_dict(checkpoint['state_dict'])
            optim_0.load_state_dict(checkpoint['opt_dict'])
            scheduler_0.load_state_dict(checkpoint['scheduler_dict'])

            np_rndstate_chkpt = checkpoint['np_rndstate']
            trch_rndstate_chkpt = checkpoint['trch_rndstate']

            np.random.set_state(np_rndstate_chkpt)
            torch.set_rng_state(trch_rndstate_chkpt)

            epoch_resume = checkpoint["epoch"] + 1
            bestLoss = checkpoint["best_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()

                if device.type == "cuda":
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]

                if phase == "train":                    
                    
                    dataloader_lb_seg = torch.utils.data.DataLoader(
                        ds['lb_seg'], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                    dataloader_lb_cyc = torch.utils.data.DataLoader(
                        ds['lb_cyc'], batch_size=1, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                    dataloader_unlb_cyc = torch.utils.data.DataLoader(
                        ds['unlb_cyc'], batch_size=1, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                    
                    loss, loss_seg, lrgdice, smldice, loss_cyc, large_inter_0, large_union_0, small_inter_0, small_union_0  = echonet.utils.seg_cycle.run_epoch_ssl( model_0, 
                                                                    dataloader_lb_seg, 
                                                                    dataloader_lb_cyc, 
                                                                    dataloader_unlb_cyc, 
                                                                    phase == "train", 
                                                                    optim_0,
                                                                    batch_size, 
                                                                    device, 
                                                                    output, 
                                                                    phase, 
                                                                    mean,
                                                                    std,
                                                                    epoch,
                                                                    chunk_size = chunk_size,
                                                                    cyc_off = cyc_off,
                                                                    target_region = target_region,
                                                                    temperature = temperature,
                                                                    val_chunk = val_chunk,
                                                                    loss_cyc_w = loss_cyc_w,
                                                                    css_strtup = css_strtup
                                                                    ) 


                    overall_dice_0 = 2 * (large_inter_0.sum() + small_inter_0.sum()) / (large_union_0.sum() + large_inter_0.sum() + small_union_0.sum() + small_inter_0.sum())
                    large_dice_0 = 2 * large_inter_0.sum() / (large_union_0.sum() + large_inter_0.sum())
                    small_dice_0 = 2 * small_inter_0.sum() / (small_union_0.sum() + small_inter_0.sum())
                    
                    f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                phase,
                                                                loss,
                                                                loss_seg,
                                                                loss_cyc,
                                                                overall_dice_0,
                                                                large_dice_0,
                                                                small_dice_0,
                                                                time.time() - start_time,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size))
                    f.flush()

                else:
                    dataloader_lb_seg = torch.utils.data.DataLoader(
                        ds['lb_seg'], batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                    dataloader_lb_cyc = torch.utils.data.DataLoader(
                        ds['lb_cyc'], batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                    dataloader_unlb_cyc = torch.utils.data.DataLoader(
                        ds['unlb_cyc'], batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                    
                    
                    loss, loss_seg_val, lrgdice_val, smldice_val, loss_cyc_val, large_inter_0, large_union_0, small_inter_0, small_union_0 = echonet.utils.seg_cycle.run_epoch_ssl( model_0, 
                                                                    dataloader_lb_seg, 
                                                                    dataloader_lb_cyc, 
                                                                    dataloader_unlb_cyc, 
                                                                    phase == "train", 
                                                                    optim_0,
                                                                    batch_size, 
                                                                    device,
                                                                    output, 
                                                                    phase,
                                                                    mean,
                                                                    std,
                                                                    epoch,
                                                                    chunk_size = chunk_size,
                                                                    cyc_off = cyc_off,
                                                                    target_region = target_region,
                                                                    temperature = temperature,
                                                                    val_chunk = val_chunk,
                                                                    loss_cyc_w = loss_cyc_w,
                                                                    css_strtup = css_strtup
                                                                    )  

                    overall_dice_0 = 2 * (large_inter_0.sum() + small_inter_0.sum()) / (large_union_0.sum() + large_inter_0.sum() + small_union_0.sum() + small_inter_0.sum())
                    large_dice_0 = 2 * large_inter_0.sum() / (large_union_0.sum() + large_inter_0.sum())
                    small_dice_0 = 2 * small_inter_0.sum() / (small_union_0.sum() + small_inter_0.sum())

                    f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                phase,
                                                                loss,
                                                                loss_seg_val,
                                                                loss_cyc_val,
                                                                overall_dice_0,
                                                                large_dice_0,
                                                                small_dice_0,
                                                                time.time() - start_time,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size))
                    
                    
                    
                    f.flush()
                  
            
            scheduler_0.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model_0.state_dict(),
                'best_loss': bestLoss,
                'loss': loss,
                'opt_dict': optim_0.state_dict(),
                'scheduler_dict': scheduler_0.state_dict(),
                'np_rndstate': np.random.get_state(),
                'trch_rndstate': torch.get_rng_state()                
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if loss_seg_val < bestLoss:
                print("saved best because {} < {}".format(loss_seg_val, bestLoss))
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = loss_seg_val

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model_0.load_state_dict(checkpoint['state_dict'])

            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["loss"], checkpoint["epoch"]))
            f.flush()

        if run_test:
            for split in ["val", "test"]:
                if reduced_set:
                    if split == "train":
                        dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs_seg, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 2)
                    else:
                        dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs_seg, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel))
                else:
                    dataset = echonet.datasets.Echo(root=data_dir, split=split, **kwargs_seg, ssl_postfix="")
                
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                loss, large_inter, large_union, small_inter, small_union = echonet.utils.seg_cycle.run_epoch(model_0, dataloader, False, None, device)

                overall_dice = 2 * (large_inter + small_inter) / (large_union + large_inter + small_union + small_inter)
                large_dice = 2 * large_inter / (large_union + large_inter)
                small_dice = 2 * small_inter / (small_union + small_inter)
                with open(os.path.join(output, "{}_dice.csv".format(split)), "w") as g:
                    g.write("Filename, Overall, Large, Small\n")
                    for (filename, overall, large, small) in zip(dataset.fnames, overall_dice, large_dice, small_dice):
                        g.write("{},{},{},{}\n".format(filename, overall, large, small))

                f.write("{} dice (overall): {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(np.concatenate((large_inter, small_inter)), np.concatenate((large_union, small_union)), echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (large):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(large_inter, large_union, echonet.utils.dice_similarity_coefficient)))
                f.write("{} dice (small):   {:.4f} ({:.4f} - {:.4f})\n".format(split, *echonet.utils.bootstrap(small_inter, small_union, echonet.utils.dice_similarity_coefficient)))
                f.flush()

    if run_inference:
        if run_inference == "all":
            run_inference_range = ['train', 'val', 'test']
        else:
            run_inference_range = [run_inference]

        for run_inference_itr in run_inference_range:
            if run_inference_itr != "train" or True:
                dataset = echonet.datasets.Echo(root=data_dir, split=run_inference_itr,
                                                    target_type=["Filename", "LargeIndex", "SmallIndex"],  # Need filename for saving, and human-selected frames to annotate
                                                    mean=mean, std=std,  # Normalization
                                                    length=None, max_length=None, period=1  # Take all frames
                                                    )
            else:
                if reduced_set:
                    dataset = echonet.datasets.Echo(root=data_dir, split=run_inference_itr,
                                                    target_type=["Filename", "LargeIndex", "SmallIndex"],  # Need filename for saving, and human-selected frames to annotate
                                                    mean=mean, std=std,  # Normalization
                                                    ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 1, ssl_mult = 1,
                                                    length=None, max_length=None, period=1  # Take all frames
                                                    )
                else:
                    dataset = echonet.datasets.Echo(root=data_dir, split=run_inference_itr,
                                                    target_type=["Filename", "LargeIndex", "SmallIndex"],  # Need filename for saving, and human-selected frames to annotate
                                                    mean=mean, std=std,  # Normalization
                                                    length=None, max_length=None, period=1  # Take all frames
                                                    )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=num_workers, shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)

            output_dir = os.path.join(output, "{}_infer_cmpct".format(run_inference_itr))

            os.makedirs(output_dir, exist_ok = True)
            
            checkpoint = torch.load(os.path.join(output, "best.pt"))

            model_0.load_state_dict(checkpoint['state_dict'])
            
            model_0.eval()

            with torch.no_grad():
                for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
                    # Run segmentation model on blocks of frames one-by-one
                    # The whole concatenated video may be too long to run together

                    print(os.path.join(output_dir, "{}_{}.npy".format(filenames[-1].replace(".avi", ""), length[-1] - 1)))

                    
                    if os.path.isfile(os.path.join(output_dir, "{}.npy".format(filenames[-1].replace(".avi", "")))):
                        # print("already exists")
                        continue

                    y = np.concatenate([model_0(x[i:(i + batch_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], batch_size)])
                    
                    y_idx = 0
                    for batch_idx in range(len(filenames)):
                        filename_itr = filenames[batch_idx] 

                        logit = y[y_idx:y_idx + length[batch_idx], 0, :, :]

                        logit_out_path = os.path.join(output_dir, "{}.npy".format(filename_itr.replace(".avi", "")))
                        np.save(logit_out_path, logit)
                        y_idx = y_idx + length[batch_idx]

            pass






def run_epoch_ssl(model_0, 
                dataloader_lb_seg, 
                dataloader_lb_cyc, 
                dataloader_unlb_cyc, 
                train, 
                optim_0, 
                batch_size, 
                device, 
                output, 
                phase, 
                mean, 
                std, 
                epoch, 
                chunk_size = 3, 
                cyc_off = 2, 
                target_region = 15, 
                temperature = 10, 
                val_chunk = 40, 
                loss_cyc_w = 1,
                css_strtup = 0
                ):


    n = 0
    n_seg = 0

    total = 0
    total_cyc = 0

    total_seg = 0

    model_0.train(train)
    output_dir = os.path.join(output, "{}_feat_comp".format(phase))
    os.makedirs(output_dir, exist_ok = True)

    large_inter_0 = 0
    large_union_0 = 0
    small_inter_0 = 0
    small_union_0 = 0
    large_inter_list_0 = []
    large_union_list_0 = []
    small_inter_list_0 = []
    small_union_list_0 = []

    torch.set_grad_enabled(train)

    total_itr_num = len(dataloader_lb_seg)

    dataloader_lb_seg_itr = iter(dataloader_lb_seg)
    dataloader_unlb_cyc_itr = iter(dataloader_unlb_cyc)

    for train_iter in range(total_itr_num):

        #### Supervised segmentation 
        _, (large_frame, small_frame, large_trace, small_trace) = dataloader_lb_seg_itr.next()

        large_frame = large_frame.to(device)
        large_trace = large_trace.to(device)

        small_frame = small_frame.to(device)
        small_trace = small_trace.to(device)

        if not train:
            with torch.no_grad():
                y_large_0 = model_0(large_frame)["out"]
        else:
            y_large_0 = model_0(large_frame)["out"]

        loss_large_0 = torch.nn.functional.binary_cross_entropy_with_logits(y_large_0[:, 0, :, :], large_trace, reduction="sum")
        # Compute pixel intersection and union between human and computer segmentations
        large_inter_0 += np.logical_and(y_large_0[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
        large_union_0 += np.logical_or(y_large_0[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
        large_inter_list_0.extend(np.logical_and(y_large_0[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
        large_union_list_0.extend(np.logical_or(y_large_0[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

        y_small_0 = model_0(small_frame)["out"]
        loss_small_0 = torch.nn.functional.binary_cross_entropy_with_logits(y_small_0[:, 0, :, :], small_trace, reduction="sum")
        # Compute pixel intersection and union between human and computer segmentations
        small_inter_0 += np.logical_and(y_small_0[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
        small_union_0 += np.logical_or(y_small_0[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
        small_inter_list_0.extend(np.logical_and(y_small_0[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
        small_union_list_0.extend(np.logical_or(y_small_0[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

        loss_seg = (loss_large_0 + loss_small_0) / 2

        loss_seg_item = loss_seg.item()
        large_trace_size = large_trace.size(0)
        total_seg += loss_seg_item * large_trace.size(0)


        
        ### CSS training
        X_raw, target, target_iekd, start, video_path, i1, j1 = dataloader_unlb_cyc_itr.next()
        X_bfcwh = X_raw.permute(0,2,1,3,4)
        X_segfeed = X_bfcwh.reshape(-1, X_bfcwh.shape[2], X_bfcwh.shape[3], X_bfcwh.shape[4])

        ####### get feature output
        if not train:
            with torch.no_grad():
                feat_out = model_0(X_segfeed)['x_layer4'].sum(dim=(2,3))
        else:
            feat_out = model_0(X_segfeed)['x_layer4'].sum(dim=(2,3))

        
        feat_out_query = feat_out[:target_region] # Template region P
        feat_out_query_cyc = feat_out[cyc_off:target_region] # Template region with offset
        feat_out_key = feat_out[target_region:] # Search region Q

        target_strtpt = np.random.choice(target_region - (chunk_size + cyc_off) + 1)  ## choosing p*
        target_strtpt_1ht = torch.eye(target_region - (chunk_size + cyc_off) + 1)[target_strtpt] 
        target_strtpt_1ht = target_strtpt_1ht.to(device)
        
        query_feat = feat_out_query[target_strtpt:target_strtpt + chunk_size, ...]  ### choosing E^p*

        key_size = feat_out_key.shape[0] 
        feat_size = feat_out.shape[1]

        ### feature-wise distance calculation 
        dist_mat = feat_out_key.unsqueeze(1).repeat((1,chunk_size, 1)) - query_feat.unsqueeze(1).transpose(0,1).repeat(key_size, 1, 1) 
        dist_mat_sq = dist_mat.pow(2) 
        dist_mat_sq_ftsm = dist_mat_sq.sum(dim = -1)

        ### distance calculation per phase
        indices_ftsm = torch.arange(chunk_size)
        gather_indx_ftsm = torch.arange(key_size).view((key_size, 1)).repeat((1,chunk_size)) 
        gather_indx_shft_ftsm = (gather_indx_ftsm + indices_ftsm) % (key_size) ### gets a index corresponding to the feature vectors included in each phase
        gather_indx_shft_ftsm = gather_indx_shft_ftsm.to(device)
        dist_mat_sq_shft_ftsm = torch.gather(dist_mat_sq_ftsm, 0, gather_indx_shft_ftsm)[:key_size - (chunk_size + cyc_off) + 1] ### gathers the feature-wise distance values to calculate the distance for the phase
        dist_mat_sq_total_ftsm = dist_mat_sq_shft_ftsm.sum(dim=(1))   
        
        ### calculating similarity value
        similarity = - dist_mat_sq_total_ftsm
        similarity_averaged = similarity / feat_size / chunk_size * temperature
        alpha_raw = torch.nn.functional.softmax(similarity_averaged, dim = 0)
        alpha_weights = alpha_raw.unsqueeze(1).unsqueeze(1).repeat([1, chunk_size, feat_size])
        

        #### calculate shifted phase values 
        indices_beta = torch.arange(chunk_size).view((1, chunk_size, 1)).repeat((key_size,1, feat_size))
        gather_indx_beta = torch.arange(key_size).view((key_size, 1, 1)).repeat((1,chunk_size, feat_size))
        gather_indx_alpha_shft = (gather_indx_beta + indices_beta) % (key_size)
        gather_indx_alpha_shft = gather_indx_alpha_shft.to(device)
        feat_out_key_beta = torch.gather(feat_out_key.unsqueeze(1).repeat(1, chunk_size, 1), 0, gather_indx_alpha_shft)[cyc_off:key_size - chunk_size + 1] 

        ### calculate \tilde{E}^{q+c}
        weighted_features = alpha_weights * feat_out_key_beta 
        weighted_features_averaged = weighted_features.sum(dim=0)


        #### match back to template region and find distance value
        q_dist_mat = feat_out_query_cyc.unsqueeze(1).repeat((1,chunk_size, 1)) - weighted_features_averaged.unsqueeze(1).transpose(0,1).repeat((target_region - cyc_off), 1, 1)
        q_dist_mat_sq = q_dist_mat.pow(2)
        q_dist_mat_sq_ftsm = q_dist_mat_sq.sum(dim = -1)

        indices_query_ftsm = torch.arange(chunk_size)
        gather_indx_query_ftsm = torch.arange(target_region - cyc_off).view((target_region - cyc_off, 1)).repeat((1,chunk_size))
        gather_indx_query_shft_ftsm = (gather_indx_query_ftsm + indices_query_ftsm) % (target_region - cyc_off)
        gather_indx_query_shft_ftsm = gather_indx_query_shft_ftsm.to(device)
        q_dist_mat_sq_shft_ftsm = torch.gather(q_dist_mat_sq_ftsm, 0, gather_indx_query_shft_ftsm)[:(target_region - cyc_off) - chunk_size + 1]
        q_dist_mat_sq_total_ftsm = q_dist_mat_sq_shft_ftsm.sum(dim=(1))

        ### calculate similarity value
        q_similarity = - q_dist_mat_sq_total_ftsm
        q_similarity_averaged = q_similarity / feat_size / chunk_size * temperature

        ### calculate cross-entropy loss
        frm_prd = torch.argmax(q_similarity_averaged)
        frm_lb = torch.argmax(target_strtpt_1ht)

        loss_cyc_raw = torch.nn.functional.cross_entropy(q_similarity_averaged.unsqueeze(0), frm_lb.unsqueeze(0))
        loss_cyc_wght = loss_cyc_raw * loss_cyc_w
        
        loss_cyc_raw_item = loss_cyc_raw.item()
        total_cyc += loss_cyc_raw_item


        if train:
            if epoch < css_strtup:
                loss_total = loss_seg
            else:
                loss_total = loss_seg + loss_cyc_wght
            optim_0.zero_grad()
            loss_total.backward()
            optim_0.step()
            
        
        loss_total_item = loss_seg_item + loss_cyc_raw_item * loss_cyc_w
        
        total += loss_total_item

        n += 1
        n_seg += large_trace_size 
    

        # Show info on process bar
        if train_iter % 5 == 0:
            print("Itr trainphase {} - {}/{} - ttl {:.4f} ({:.4f})  seg {:.4f} ({:.4f})  dlrg {:.4f}  dsml {:.4f}  cyc {:.4f} ({:.4f}) ".format(
                train, 
                train_iter, 
                total_itr_num,
                total / n_seg ,  # total
                loss_total_item, # total_item
                total_seg / n_seg / 112 / 112 ,  # total seg
                loss_seg_item,  # seg item 
                2 * large_inter_0 / (large_union_0 + large_inter_0 + 0.000001), 
                2 * small_inter_0 / (small_union_0 + small_inter_0 + 0.000001), 
                total_cyc / n,
                loss_cyc_raw_item
                ), flush = True)

    large_inter_list_0 = np.array(large_inter_list_0)
    large_union_list_0 = np.array(large_union_list_0)
    small_inter_list_0 = np.array(small_inter_list_0)
    small_union_list_0 = np.array(small_union_list_0)

    return (total / n_seg, 
            total_seg / n_seg / 112 / 112, 
            2 * large_inter_0 / (large_union_0 + large_inter_0 + 0.000001), 
            2 * small_inter_0 / (small_union_0 + small_inter_0 + 0.000001), 
            total_cyc / n, 
            large_inter_list_0,
            large_union_list_0,
            small_inter_list_0,
            small_union_list_0
             )




def run_epoch(model, dataloader, train, optim, device):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
    """

    total = 0.
    n = 0

    pos = 0
    neg = 0
    pos_pix = 0
    neg_pix = 0

    model.train(train)

    large_inter = 0
    large_union = 0
    small_inter = 0
    small_union = 0
    large_inter_list = []
    large_union_list = []
    small_inter_list = []
    small_union_list = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (_, (large_frame, small_frame, large_trace, small_trace)) in dataloader:
                # Count number of pixels in/out of human segmentation
                pos += (large_trace == 1).sum().item()
                pos += (small_trace == 1).sum().item()
                neg += (large_trace == 0).sum().item()
                neg += (small_trace == 0).sum().item()

                # Count number of pixels in/out of computer segmentation
                pos_pix += (large_trace == 1).sum(0).to("cpu").detach().numpy()
                pos_pix += (small_trace == 1).sum(0).to("cpu").detach().numpy()
                neg_pix += (large_trace == 0).sum(0).to("cpu").detach().numpy()
                neg_pix += (small_trace == 0).sum(0).to("cpu").detach().numpy()

                # Run prediction for diastolic frames and compute loss
                large_frame = large_frame.to(device)
                large_trace = large_trace.to(device)
                y_large = model(large_frame)["out"]
                loss_large = torch.nn.functional.binary_cross_entropy_with_logits(y_large[:, 0, :, :], large_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                large_inter += np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_union += np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                large_inter_list.extend(np.logical_and(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                large_union_list.extend(np.logical_or(y_large[:, 0, :, :].detach().cpu().numpy() > 0., large_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Run prediction for systolic frames and compute loss
                small_frame = small_frame.to(device)
                small_trace = small_trace.to(device)
                y_small = model(small_frame)["out"]
                loss_small = torch.nn.functional.binary_cross_entropy_with_logits(y_small[:, 0, :, :], small_trace, reduction="sum")
                # Compute pixel intersection and union between human and computer segmentations
                small_inter += np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_union += np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum()
                small_inter_list.extend(np.logical_and(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))
                small_union_list.extend(np.logical_or(y_small[:, 0, :, :].detach().cpu().numpy() > 0., small_trace[:, :, :].detach().cpu().numpy() > 0.).sum((1, 2)))

                # Take gradient step if training
                loss = (loss_large + loss_small) / 2
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                # Accumulate losses and compute baselines
                total += loss.item()
                n += large_trace.size(0)
                p = pos / (pos + neg)
                p_pix = (pos_pix + 1) / (pos_pix + neg_pix + 2)

                # Show info on process bar
                pbar.set_postfix_str("{:.4f} ({:.4f}) / {:.4f} {:.4f}, {:.4f}, {:.4f}".format(total / n / 112 / 112, loss.item() / large_trace.size(0) / 112 / 112, -p * math.log(p) - (1 - p) * math.log(1 - p), (-p_pix * np.log(p_pix) - (1 - p_pix) * np.log(1 - p_pix)).mean(), 2 * large_inter / (large_union + large_inter), 2 * small_inter / (small_union + small_inter)))
                pbar.update()

    large_inter_list = np.array(large_inter_list)
    large_union_list = np.array(large_union_list)
    small_inter_list = np.array(small_inter_list)
    small_union_list = np.array(small_union_list)

    return (total / n / 112 / 112,
            large_inter_list,
            large_union_list,
            small_inter_list,
            small_union_list,
            )


def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i

