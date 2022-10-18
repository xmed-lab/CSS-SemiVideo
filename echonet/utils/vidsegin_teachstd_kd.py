"""Teacher Student Distillation"""

import math
import os
import time
import shutil
import datetime
import pandas as pd
import cv2

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm
import subprocess

import echonet
import echonet.models

@click.command("vidsegin_teachstd_kd")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--task", type=str, default="EF")
@click.option("--model_name", type=click.Choice(['mc3_18', 'r2plus1d_18', "r2plus1d_18_segin", 'r3d_18', 'r2plus1d_18_ncor']),
    default="r2plus1d_18")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--weights_0", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=False)
@click.option("--num_epochs", type=int, default=30)  
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=15)
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)
@click.option("--full_test/--quick_test", default=False)
@click.option("--val_samp", type=int, default=3)
@click.option("--reduced_set/--full_set", default=True)
@click.option("--rd_label", type=int, default=920)
@click.option("--rd_unlabel", type=int, default=6440)
@click.option("--max_block", type=int, default=20)
@click.option("--segsource", type=str, default=None)
@click.option("--w_unlb", type=float, default=2.5)
@click.option("--batch_size_unlb", type=int, default=10)
@click.option("--notcamus/--camus", default=True)

def run(
    data_dir=None,
    output=None,
    task="EF",
    model_name="r2plus1d_18",
    pretrained=True,
    weights=None,
    weights_0=None,
    run_test=False,
    num_epochs=30,
    lr=1e-4,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,
    num_train_patients=None,
    num_workers=4,
    batch_size=20,
    device=None,
    seed=0,
    full_test = False,
    val_samp = 3,
    reduced_set = True,
    rd_label = 920,
    rd_unlabel = 6440,
    max_block = 20,
    segsource = None,    
    w_unlb = 2.5,
    batch_size_unlb=10,
    notcamus = True
):
    
    assert segsource, "function needs segsource option"


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
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(model_name, frames, period, "pretrained" if pretrained else "random"))
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

    model_0 = echonet.models.rnet2dp1.r2plus1d_18_kd(pretrained=pretrained)
    model_0.fc = torch.nn.Linear(model_0.fc.in_features, 1)
    model_0.fc.bias.data[0] = 55.6        

    model_ref = echonet.models.rnet2dp1.r2plus1d_18(pretrained=pretrained)

    model_0.stem = torch.nn.Sequential(
    torch.nn.Conv3d(4, 45, kernel_size=(1, 7, 7),
                stride=(1, 2, 2), padding=(0, 3, 3),
                bias=False),
    torch.nn.BatchNorm3d(45),
    torch.nn.ReLU(inplace=True),
    torch.nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                stride=(1, 1, 1), padding=(1, 0, 0),
                bias=False),
    torch.nn.BatchNorm3d(64),
    torch.nn.ReLU(inplace=True))

    for weight_itr in range(1,6):
        model_0.stem[weight_itr].load_state_dict(model_ref.stem[weight_itr].state_dict())

    model_0.stem[0].weight.data[:,:3,:,:,:] = model_ref.stem[0].weight.data[:,:,:,:,:]
    model_0.stem[0].weight.data[:,3,:,:,:] = torch.tensor(np.random.uniform(low = -1, high = 1, size = model_0.stem[0].weight.data[:,3,:,:,:].shape)).float()
        
    model_0 = torch.nn.DataParallel(model_0)

    model_0.to(device) 


    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    ### we initialize teacher and student weights.
    if weights_0 is not None:
        checkpoint_0 = torch.load(weights_0)
        if checkpoint_0.get("state_dict_0"):
            ## initialize teacher weights
            print("loading from state_dict_0")
            model_0.load_state_dict(checkpoint_0['state_dict_0'])

            ## initialize student weights where transferable to speed up training
            state_dict = checkpoint_0['state_dict_0']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            
            model = echonet.models.rnet2dp1.r2plus1d_18_kd(pretrained=pretrained)
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
            model.fc.bias.data[0] = 55.6        

            model.stem = torch.nn.Sequential(
                torch.nn.Conv3d(4, 45, kernel_size=(1, 7, 7),
                            stride=(1, 2, 2), padding=(0, 3, 3),
                            bias=False),
                torch.nn.BatchNorm3d(45),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                            stride=(1, 1, 1), padding=(1, 0, 0),
                            bias=False),
                torch.nn.BatchNorm3d(64),
                torch.nn.ReLU(inplace=True))

            model.load_state_dict(new_state_dict)

            model.stem = torch.nn.Sequential(
                torch.nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                            stride=(1, 2, 2), padding=(0, 3, 3),
                            bias=False),
                torch.nn.BatchNorm3d(45),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                            stride=(1, 1, 1), padding=(1, 0, 0),
                            bias=False),
                torch.nn.BatchNorm3d(64),
                torch.nn.ReLU(inplace=True))

            for weight_itr in range(1,6):
                model.stem[weight_itr].load_state_dict(model_0.module.stem[weight_itr].state_dict())

            model.stem[0].weight.data[:,:3,:,:,:] = model_0.module.stem[0].weight.data[:,:3,:,:,:]

            model = torch.nn.DataParallel(model)   
            model.to(device)

        elif checkpoint_0.get("state_dict"):
            ## initialize teacher weights
            print("loading from state_dict")
            model_0.load_state_dict(checkpoint_0['state_dict'])

            ## initialize student weights where transferable to speed up training
            state_dict = checkpoint_0['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            
            model = echonet.models.rnet2dp1.r2plus1d_18_kd(pretrained=pretrained)
            model.fc = torch.nn.Linear(model.fc.in_features, 1)
            model.fc.bias.data[0] = 55.6        

            model.stem = torch.nn.Sequential(
                torch.nn.Conv3d(4, 45, kernel_size=(1, 7, 7),
                            stride=(1, 2, 2), padding=(0, 3, 3),
                            bias=False),
                torch.nn.BatchNorm3d(45),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                            stride=(1, 1, 1), padding=(1, 0, 0),
                            bias=False),
                torch.nn.BatchNorm3d(64),
                torch.nn.ReLU(inplace=True))

            model.load_state_dict(new_state_dict)

            model.stem = torch.nn.Sequential(
                torch.nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                            stride=(1, 2, 2), padding=(0, 3, 3),
                            bias=False),
                torch.nn.BatchNorm3d(45),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                            stride=(1, 1, 1), padding=(1, 0, 0),
                            bias=False),
                torch.nn.BatchNorm3d(64),
                torch.nn.ReLU(inplace=True))

            for weight_itr in range(1,6):
                model.stem[weight_itr].load_state_dict(model_0.module.stem[weight_itr].state_dict())

            model.stem[0].weight.data[:,:3,:,:,:] = model_0.module.stem[0].weight.data[:,:3,:,:,:]

            model = torch.nn.DataParallel(model)   
            model.to(device)
        else:
            assert 1==2, "missing key"


    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    optim_0 = torch.optim.SGD(model_0.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler_0 = torch.optim.lr_scheduler.StepLR(optim_0, lr_step_period)


    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo_tskd(root=data_dir, split="train")) 
    print("mean std", mean, std)
    kwargs = {"target_type": ["EF", "IEKD"],
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    kwargs_testall = {"target_type": "EF",
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    dataset = {}
    dataset_trainsub = {}
    if reduced_set:
        dataset_trainsub['lb'] = echonet.datasets.Echo_tskd(root=data_dir, split="train", **kwargs, pad=12, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 1, segin_dir = "../infer_buffers/{}/train_infer_cmpct".format(segsource))
        dataset_trainsub["unlb_0"] = echonet.datasets.Echo_tskd(root=data_dir, split="train", **kwargs, pad=12, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), ssl_type = 2, segin_dir = "../infer_buffers/{}/train_infer_cmpct".format(segsource))
        dataset["train"] = dataset_trainsub
        dataset["val"] = echonet.datasets.Echo_tskd(root=data_dir, split="val", **kwargs, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), segin_dir = "../infer_buffers/{}/val_infer_cmpct".format(segsource))
    else:
        dataset["train"] = echonet.datasets.Echo_tskd(root=data_dir, split="train", **kwargs, pad=12, ssl_postfix="", segin_dir = "../infer_buffers/{}/train_infer_cmpct".format(segsource))
        dataset["val"] = echonet.datasets.Echo_tskd(root=data_dir, split="val", **kwargs, ssl_postfix="", segin_dir = "../infer_buffers/{}/val_infer_cmpct".format(segsource))
    

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:

        f.write("Run timestamp: {}\n".format(bkup_tmstmp))

        epoch_resume = 0
        bestLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'], strict = False)
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])

            model_0.load_state_dict(checkpoint['state_dict_0'], strict = False)
            optim_0.load_state_dict(checkpoint['opt_dict_0'])
            scheduler_0.load_state_dict(checkpoint['scheduler_dict_0'])

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

                if phase == "train":
                    ds_lb = dataset[phase]['lb']
                    dataloader_lb = torch.utils.data.DataLoader(
                        ds_lb, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)

                    ds_unlb_0 = dataset[phase]['unlb_0']
                    dataloader_unlb_0 = torch.utils.data.DataLoader(
                        ds_unlb_0, batch_size=batch_size_unlb, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"), worker_init_fn=worker_init_fn)

                    
                    total, total_reg, total_reg_reg, total_reg_unlb, yhat, y = echonet.utils.vidsegin_teachstd_kd.run_epoch(model = model, 
                                                                                                                        model_0 = model_0, 
                                                                                                                        dataloader = dataloader_lb, 
                                                                                                                        dataloader_unlb_0 = dataloader_unlb_0,
                                                                                                                        train = phase == "train", 
                                                                                                                        optim = optim, 
                                                                                                                        optim_0 = optim_0, 
                                                                                                                        device = device, 
                                                                                                                        w_unlb = w_unlb)


                    r2_value = sklearn.metrics.r2_score(y, yhat)

                    f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                                phase,
                                                                total,
                                                                total_reg_reg,
                                                                r2_value,
                                                                total_reg_unlb,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size))
                    f.flush()
                
                    # print("successful run until exit")
                    # exit()
                
                else:
                    ### for validation 
                    ### store seeds 
                    np_rndstate = np.random.get_state()
                    trch_rndstate = torch.get_rng_state()

                    r2_track = []
                    lossreg_track = []


                    for val_samp_itr in range(val_samp):
                        
                        print("running validation batch for seed =", val_samp_itr)

                        np.random.seed(val_samp_itr)
                        torch.manual_seed(val_samp_itr)
    
                        ds = dataset[phase]
                        dataloader = torch.utils.data.DataLoader(
                            ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))
                                                                                                                        
                        
                        total, total_reg, total_reg_reg, total_reg_unlb, yhat, y = echonet.utils.vidsegin_teachstd_kd.run_epoch(model = model, 
                                                                                                                            model_0 = model_0, 
                                                                                                                            dataloader = dataloader, 
                                                                                                                            dataloader_unlb_0 = None, 
                                                                                                                            train = phase == "train", 
                                                                                                                            optim = optim, 
                                                                                                                            optim_0 = optim_0, 
                                                                                                                            device = device, 
                                                                                                                            w_unlb = w_unlb)


                        r2_track.append(sklearn.metrics.r2_score(y, yhat))
                        lossreg_track.append(total_reg_reg)


                    r2_value = np.average(np.array(r2_track))
                    lossreg = np.average(np.array(lossreg_track))

                    f.write("{},{},{},{},{},{},{},{},{},{}".format(epoch,
                                                                phase,
                                                                lossreg,
                                                                r2_value,
                                                                total_reg_unlb,
                                                                time.time() - start_time,
                                                                y.size,
                                                                sum(torch.cuda.max_memory_allocated() for i in range(torch.cuda.device_count())),
                                                                sum(torch.cuda.max_memory_reserved() for i in range(torch.cuda.device_count())),
                                                                batch_size))
            
                    for trck_write in range(len(r2_track)):
                        f.write(",{}".format(r2_track[trck_write]))

                    for trck_write in range(len(lossreg_track)):
                        f.write(",{}".format(lossreg_track[trck_write]))


                    f.write("\n")
                    f.flush()
                    
                    np.random.set_state(np_rndstate)
                    torch.set_rng_state(trch_rndstate)

            
            scheduler.step()
            scheduler_0.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'state_dict_0': model_0.state_dict(),
                'period': period,
                'frames': frames,
                'best_loss': bestLoss,
                'loss': lossreg,
                'r2': r2_value,
                'opt_dict': optim.state_dict(),
                'opt_dict_0': optim_0.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
                'scheduler_dict_0': scheduler_0.state_dict(),
                'np_rndstate': np.random.get_state(),
                'trch_rndstate': torch.get_rng_state()
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))

            #### save based on reg loss
            if lossreg < bestLoss:
                print("saved best because {} < {}".format(lossreg, bestLoss))
                torch.save(save, os.path.join(output, "best.pt"))
                bestLoss = lossreg


        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'], strict = False)
            model_0.load_state_dict(checkpoint['state_dict_0'], strict = False)
            f.write("Best validation loss {} from epoch {}, R2 {}\n".format(checkpoint["loss"], checkpoint["epoch"], checkpoint["r2"]))
            f.flush()

        if run_test:
            if notcamus:
                split_list = ["test", "val"]
                # split_list = ["test"]
            else:
                split_list = ["train", "test"]

            for split in split_list:
                # Performance without test-time augmentation

                if not full_test:

                    for seed_itr in range(5):
                        np.random.seed(seed_itr)
                        torch.manual_seed(seed_itr)
                        
                        dataloader = torch.utils.data.DataLoader(
                            echonet.datasets.Echo(root=data_dir, split=split, **kwargs_testall, ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel), segin_dir = "../infer_buffers/{}/{}_infer_cmpct".format(segsource, split)),
                            batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), worker_init_fn=worker_init_fn)
                        


                        loss, yhat, y = echonet.utils.vidsegin_teachstd_kd.run_epoch_val(model = model, 
                                                                                                model_0 = model_0, 
                                                                                                dataloader = dataloader, 
                                                                                                train = False, 
                                                                                                optim = None, 
                                                                                                optim_0 = None, 
                                                                                                device = device)

                        f.write("Seed is {}\n".format(seed_itr))
                        f.write("{} - {} (one clip) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                        f.write("{} - {} (one clip) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                        f.write("{} - {} (one clip) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                        f.flush()


                else:
                    # Performance with test-time augmentation     
                    if reduced_set:           
                        ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs_testall, clips="all", ssl_postfix="_ssl_{}_{}".format(rd_label, rd_unlabel))
                    else:
                        ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs_testall, clips="all", ssl_postfix="")

                    yhat, y = echonet.utils.vidsegin_teachstd_kd.test_epoch_all(model, 
                                                                                ds, 
                                                                                False, 
                                                                                None, 
                                                                                device, 
                                                                                save_all=True, 
                                                                                block_size=batch_size, 
                                                                                run_dir = output, 
                                                                                test_val = split, 
                                                                                **kwargs)

                    f.write("Seed is {} \n".format(seed))
                    f.write("{} - {} (all clips, mod) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.r2_score)))
                    f.write("{} - {} (all clips, mod) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_absolute_error)))
                    f.write("{} - {} (all clips, mod) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y, yhat, sklearn.metrics.mean_squared_error)))))
                    f.flush()




def test_epoch_all(model, dataset, train, optim, device, save_all=False, block_size=None, run_dir = None, test_val = None, mean = None, std = None, length = None, period = None, target_type = None):
    model.train(False)

    total = 0  # total training loss
    total_reg = 0 
    total_ncor = 0

    n = 0      # number of videos processed
    s1 = 0     # sum of ground truth EF
    s2 = 0     # Sum of ground truth EF squared

    yhat = []
    y = []

    #### some params in the dataloader

    if (mean is None) or (std is None) or (length is None) or (period is None):
        assert 1==2, "missing key params"

    max_length = 250

    if run_dir:
    
        temp_savefile = os.path.join(run_dir, "temp_inference_{}.csv".format(test_val))

    with torch.set_grad_enabled(False):
        orig_filelist = dataset.fnames

        if os.path.isfile(temp_savefile):
            exist_data = pd.read_csv(temp_savefile)
            exist_file = list(exist_data['fnames'])
            target_filelist = sorted(list(set(orig_filelist) - set(exist_file)))
        else:
            target_filelist = sorted(list(orig_filelist))
            exist_data = pd.DataFrame(columns = ['fnames', 'yhat'])

        for filelistitr_idx in range(len(target_filelist)):
            filelistitr = target_filelist[filelistitr_idx]

            video_path = os.path.join(echonet.config.DATA_DIR, "Videos", filelistitr)
            ### Get data
            video = echonet.utils.loadvideo(video_path).astype(np.float32)

            if isinstance(mean, (float, int)):
                video -= mean
            else:
                video -= mean.reshape(3, 1, 1, 1)

            if isinstance(std, (float, int)):
                video /= std
            else:
                video /= std.reshape(3, 1, 1, 1)

            c, f, h, w = video.shape
            if length is None:
                # Take as many frames as possible
                length = f // period
            else:
                # Take specified number of frames
                length = length

            if max_length is not None:
                # Shorten videos to max_length
                length = min(length, max_length)

            if f < length * period:
                # Pad video with frames filled with zeros if too short
                # 0 represents the mean color (dark grey), since this is after normalization
                video = np.concatenate((video, np.zeros((c, length * period - f, h, w), video.dtype)), axis=1)
                c, f, h, w = video.shape  # pylint: disable=E0633

            start = np.arange(f - (length - 1) * period)
            #### Do looping starting from here

            reg1 = []
            n_clips = start.shape[0]
            batch = 1
            for s_itr in range(0, start.shape[0], block_size):
                print("{}, processing file {} out of {},  block {} out of {}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), filelistitr_idx, len(target_filelist), s_itr, start.shape[0]), flush=True)
                # print("s range", start[s_itr: s_itr + block_size])
                # print("frame range", s + period * np.arange(length))
                vid_samp = tuple(video[:, s + period * np.arange(length), :, :] for s in start[s_itr: s_itr + block_size])
                X1 = torch.tensor(np.stack(vid_samp))
                X1 = X1.to(device)

                if device.type == "cuda":
                    all_output = model(X1)
                else:
                    #### we only ever use cpu for testing
                    all_output = torch.ones((X1.shape[0]))

                reg1.append(all_output[0].detach().cpu().numpy())

            reg1 = np.vstack(reg1)
            reg1_mean = reg1.reshape(batch, n_clips, -1).mean(1)

            exist_data = exist_data.append({'fnames':filelistitr, 'yhat':reg1_mean[0,0]}, ignore_index=True)

            if filelistitr_idx % 20 == 0:
                exist_data.to_csv(temp_savefile, index = False)
    
    label_data_path = os.path.join(echonet.config.DATA_DIR, "FileList.csv")
    label_data = pd.read_csv(label_data_path)
    label_data_select = label_data[['FileName','EF']]
    label_data_select.columns = ['fnames','EF']
    with_predict = exist_data.merge(label_data_select, on='fnames')

    predict_out_path = os.path.join(run_dir, "{}_predictions.csv".format(test_val))
    with_predict.to_csv(predict_out_path, index=False)


    # print(with_predict)
    # exit()
    return with_predict['yhat'].to_numpy(), with_predict['EF'].to_numpy()


def run_epoch(model, 
                model_0, 
                dataloader, 
                dataloader_unlb_0, 
                train, 
                optim, 
                optim_0, 
                device, 
                save_all=False, 
                block_size=None, 
                run_dir = None, 
                test_val = None, 
                w_unlb = 0):
    

    total = 0  # total training loss
    total_reg = 0 
    total_reg_reg = 0 
    total_loss_reg_reg_unlb = 0


    n = 0      # number of videos processed
    n_frm = 0
    s1 = 0     # sum of ground truth EF
    s2 = 0     # Sum of ground truth EF squared

    yhat = []
    yhat_seg = []
    y = []
    start_frame_record = []
    vidpath_record = []

    if dataloader_unlb_0:
        dataloader_unlb_0_itr = iter(dataloader_unlb_0)

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            enum_idx = 0
            for (X_all, outcome, seg_info, start_frame, video_path, _, _) in dataloader:                   
                enum_idx = enum_idx + 1


                if not train:
                    start_frame_record.append(start_frame.view(-1).to("cpu").detach().numpy())
                    vidpath_record.append(video_path)

                y.append(outcome.detach().cpu().numpy())

                if X_all.dtype == torch.double:
                    X_all = X_all.float()

                X_noseg = X_all[:,:3,...].to(device)
                X_wseg = X_all.to(device)

                outcome = outcome.to(device)

                s1 += outcome.sum()
                s2 += (outcome ** 2).sum()                

                if dataloader_unlb_0:

                    (X_all_unlb, outcome_unlb, seg_info_unlb, start_frame_unlb, video_path_unlb, _, _) = dataloader_unlb_0_itr.next()
                    if X_all_unlb.dtype == torch.double:
                        X_all_unlb = X_all_unlb.float()

                    X_noseg_unlb = X_all_unlb[:,:3,...].to(device)
                    X_wseg_unlb = X_all_unlb.to(device)


                    if train:
                        model.train(True)
                    else:
                        model.train(False)
                    model_0.train(False)

                    if train:
                        all_output_unlb = model(X_noseg_unlb)
                    else:
                        with torch.no_grad():
                            all_output_unlb = model(X_noseg_unlb)

                    y_pred_unlb = all_output_unlb[0]
                    
                    with torch.no_grad():
                        all_output_seg_unlb = model_0(X_wseg_unlb)

                    y_pred_seg_unlb = all_output_seg_unlb[0]

                    y_pred_avg_unlb = y_pred_seg_unlb.view(-1).detach()

                    loss_reg_reg_unlb_vid = torch.nn.functional.mse_loss(y_pred_unlb.view(-1), y_pred_avg_unlb)
                    loss_reg_reg_unlb_seg = torch.nn.functional.mse_loss(y_pred_seg_unlb.view(-1), y_pred_avg_unlb)

                    loss_reg_reg_unlb = loss_reg_reg_unlb_vid / 2

                    loss_reg_reg_unlb_item = loss_reg_reg_unlb.item()

                else:
                    loss_reg_reg_unlb_item = 0


                #### train video model
                if train:
                    model.train(True)
                    # attKD.train(True)
                else:
                    model.train(False)
                    # attKD.train(False)
                model_0.train(False)

                if train:
                    all_output = model(X_noseg)
                else:
                    with torch.no_grad():
                        all_output = model(X_noseg)

                y_pred = all_output[0]                    

                with torch.no_grad():
                    all_output_seg = model_0(X_wseg)

                y_pred_seg = all_output_seg[0]


                loss_reg_reg = torch.nn.functional.mse_loss(y_pred.view(-1), outcome)
                yhat.append(y_pred.view(-1).to("cpu").detach().numpy())

                if dataloader_unlb_0:
                    loss_reg = loss_reg_reg + w_unlb * loss_reg_reg_unlb  
                else:
                    loss_reg = loss_reg_reg


                if train:
                    optim.zero_grad()
                    loss_reg.backward()
                    optim.step()

                total_reg += loss_reg.item() * outcome.size(0)
                total_reg_reg += loss_reg_reg.item() * outcome.size(0)


                loss_reg_item = loss_reg.item()
                loss_reg_reg_item = loss_reg_reg.item()


                total = total_reg
                if dataloader_unlb_0:
                    total_loss_reg_reg_unlb = total_loss_reg_reg_unlb + loss_reg_reg_unlb.item()
                    
                n += outcome.size(0)

                pbar.set_postfix_str("total {:.4f} / reg {:.4f} ({:.4f}) regrg {:.4f} ({:.4f}) regulb {:.4f} ({:.4f})".format(total / n , 
                    total_reg / n , loss_reg_item, 
                    total_reg_reg / n , loss_reg_reg_item, 
                    total_loss_reg_reg_unlb / n, loss_reg_reg_unlb_item
                    ))
                pbar.update()
    yhat_cat = np.concatenate(yhat)
    y = np.concatenate(y)

    
    return (total / n , 
            total_reg / n , 
            total_reg_reg / n , 
            total_loss_reg_reg_unlb / n , 
            yhat_cat, y)





def run_epoch_val(model, model_0, dataloader, train, optim, optim_0, device, save_all=False, block_size=None):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
        save_all (bool, optional): If True, return predictions for all
            test-time augmentations separately. If False, return only
            the mean prediction.
            Defaults to False.
        block_size (int or None, optional): Maximum number of augmentations
            to run on at the same time. Use to limit the amount of memory
            used. If None, always run on all augmentations simultaneously.
            Default is None.
    """

    total = 0  # total training loss
    total_reg = 0 
    start_frame_record = []
    vidpath_record = []
    yhat = []
    y = []

    n = 0
    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            enum_idx = 0
            for (X, outcome, start_frame, video_path, _, _) in dataloader:                   
                enum_idx = enum_idx + 1

                if not train:
                    start_frame_record.append(start_frame.view(-1).to("cpu").detach().numpy())
                    vidpath_record.append(video_path)

                y.append(outcome.detach().cpu().numpy())

                if X.dtype == torch.double:
                    X = X.float()

                X = X[:,:3,...].to(device)

                outcome = outcome.to(device)

                model.train(False)
                model_0.train(False)
                all_output = model(X)

                y_pred = all_output[0]
                
                loss_reg = torch.nn.functional.mse_loss(y_pred.view(-1), outcome)
                yhat.append(y_pred.view(-1).to("cpu").detach().numpy())

                total_reg += loss_reg.item() * outcome.size(0)
                total = total_reg
                n += outcome.size(0)

                pbar.set_postfix_str("total {:.4f}".format(total / n))
                pbar.update()

    yhat = np.concatenate(yhat)

    y = np.concatenate(y)

    
    return (total / n, yhat, y)


