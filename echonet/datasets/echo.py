"""EchoNet-Dynamic Dataset."""

import os
import collections
import pandas
import datetime

import numpy as np
import skimage.draw
import torchvision
import echonet

from scipy.special import expit


class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 ssl_type = 0,
                 ssl_postfix = "",
                 ssl_mult = 1,
                 ssl_edesonly = True,
                 noise=None,
                 target_transform=None,
                 external_test_location=None,
                 mask_gen = False,
                 mask_prop_range=(0.25, 0.5),
                 mask_n_boxes=3,
                 mask_random_aspect_ratio=True,
                 mask_prop_by_area=True,
                 mask_within_bounds=True,
                 mask_invert=True,
                 segin_dir = None,
                 digiseg = False
                 ):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.ssl_type = ssl_type
        self.ssl_postfix = ssl_postfix
        self.ssl_mult = ssl_mult

        self.ssl_edesonly = ssl_edesonly

        self.mask_gen = mask_gen
        self.prop_range = mask_prop_range
        self.n_boxes = mask_n_boxes
        self.random_aspect_ratio = mask_random_aspect_ratio
        self.prop_by_area = mask_prop_by_area
        self.within_bounds = mask_within_bounds
        self.invert = mask_invert

        self.segin_dir = segin_dir
        self.digiseg = digiseg

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            print("Using data file from ", os.path.join(self.root, "FileList{}.csv".format(self.ssl_postfix)))

            with open(os.path.join(self.root, "FileList{}.csv".format(self.ssl_postfix))) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]


            if self.ssl_type == 1:
                assert self.split == "TRAIN", "subset selection only for train"
                #### labeled training
                data = data[data["SSL_SPLIT"] == "LABELED"]
                print("Using SSL_SPLIT Labeled, total samples", len(data))
                #### need to double/multiply the dataset 
                data_columns = data.columns
                data = pandas.DataFrame(np.repeat(data.values,self.ssl_mult,axis=0))
                data.columns = data_columns
                print("data after duplicates:", len(data))

            elif self.ssl_type == 2:
                assert self.split == "TRAIN", "subset selection only for train"
                ### unlableled training
                data = data[data["SSL_SPLIT"] == "UNLABELED"]
                print("Using SSL_SPLIT unlabeled, total samples", len(data))
                data_columns = data.columns
                data = pandas.DataFrame(np.repeat(data.values,self.ssl_mult,axis=0))
                data.columns = data_columns
                print("data after duplicates:", len(data))

            elif self.ssl_type == 0:
                print("Using SSL_SPLIT ALL, total samples", len(data))
                pass
            else:
                assert 1==2, "invalid option for ssl_type in echonet data"

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            # print(self.fnames)
            # self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix,  
            # print(self.fnames)

            self.outcome = data.values.tolist()


            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            # print(self.fnames)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    filename = filename + ".avi"
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
        # print(self.fnames)

    def __getitem__(self, index):
        # Find filename of video

        if self.split == "EXTERNAL_TEST":
            video_path = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video_path = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video_path = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video_path).astype(np.float32)

        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        f_old = f

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)
            # print("start=", start)

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                if self.ssl_edesonly:
                    target.append(video[:, self.frames[key][-1], :, :])
                else:
                    frm_select = np.random.choice(video.shape[1])
                    target.append(video[:, frm_select, :, :])
            elif t == "SmallFrame":
                if self.ssl_edesonly:
                    target.append(video[:, self.frames[key][0], :, :])
                else:
                    frm_select = np.random.choice(video.shape[1])
                    target.append(video[:, frm_select, :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]

                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)


            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if self.mask_gen:
            n_masks = 2
            mask_shape = (112, 112)

            rng = np.random

            if self.prop_by_area:
                # Choose the proportion of each mask that should be above the threshold
                mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

                # Zeros will cause NaNs, so detect and suppres them
                zero_mask = mask_props == 0.0

                if self.random_aspect_ratio:
                    y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                    x_props = mask_props / y_props
                else:
                    y_props = x_props = np.sqrt(mask_props)
                fac = np.sqrt(1.0 / self.n_boxes)
                y_props *= fac
                x_props *= fac

                y_props[zero_mask] = 0
                x_props[zero_mask] = 0
            else:
                if self.random_aspect_ratio:
                    y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                    x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                else:
                    x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                fac = np.sqrt(1.0 / self.n_boxes)
                y_props *= fac
                x_props *= fac


            sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])  ### the [None, None, :] expands the dimension to (1,1,2)

            # print("not too sure whats going on here...", sizes.shape)
            # print("np.array(mask_shape)[None, None, :]", np.array(mask_shape)[None, None, :])
            # a = np.array(mask_shape)[None, None, :]
            # print("shape of a = np.array(mask_shape)[None, None, :]", a.shape)
            # print("np.stack([y_props, x_props], axis=2)", np.stack([y_props, x_props], axis=2).shape)
            # print("np.stack([y_props, x_props], axis=2) * np.array(mask_shape)", np.stack([y_props, x_props], axis=2) * np.array(mask_shape))
            # exit()

            if self.within_bounds:
                positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
                rectangles = np.append(positions, positions + sizes, axis=2)
                
            else:
                centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
                rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

            if self.invert:
                masks = np.zeros((n_masks, 1) + mask_shape)  ### gives a zeros matrix of shape (2,1,112,112)
            else:
                masks = np.ones((n_masks, 1) + mask_shape)
            for i, sample_rectangles in enumerate(rectangles):
                for y0, x0, y1, x1 in sample_rectangles:
                    masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
            target.append(masks)

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select clips from video
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        
        if self.segin_dir:
            seg_infer_path = os.path.join(self.segin_dir, self.fnames[index].replace(".avi", ".npy"))
            seg_infer_logits = np.load(seg_infer_path)
            if self.digiseg:
                seg_infer_probs = (seg_infer_logits > 0).astype(np.float32)
            else:
                seg_infer_probs = expit(seg_infer_logits)
            seg_infer_prob_norm = seg_infer_probs * 2 - 1

            #### check if need to append 
            if f_old < length * self.period:
                seg_infer_prob_norm = np.concatenate((seg_infer_prob_norm, np.ones((length * self.period - f_old, h, w), video[0].dtype) * -1), axis=0)

            seg_infer_prob_norm = np.expand_dims(seg_infer_prob_norm, axis=0)

            seg_infer_prob_norm_samp =  tuple(seg_infer_prob_norm[:, s + self.period * np.arange(length), :, :] for s in start)
            


        if self.clips == 1:
            video = video[0]
            if self.segin_dir:
                seg_infer_prob_norm_samp = seg_infer_prob_norm_samp[0]
                video = np.concatenate((video, seg_infer_prob_norm_samp), axis=0)
        else:
            video = np.stack(video)
            if self.segin_dir:
                seg_infer_prob_norm_samp = np.stack(seg_infer_prob_norm_samp)
                video = np.concatenate((video, seg_infer_prob_norm_samp), axis=1)


        if self.pad is not None:
            video1 = video.copy()


            c, l, h, w = video.shape

            temp1 = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp1[:, :, self.pad:-self.pad, self.pad:-self.pad] = video1


            i1, j1 = np.random.randint(0, 2 * self.pad, 2)

            video1 = temp1[:, :, i1:(i1 + h), j1:(j1 + w)]

        else:
            video1 = video.copy()
            i1 = 0
            j1 = 0
        if self.target_type == ["EF"]:
            return video1, target, start, video_path, i1, j1
        else:
            return video1, target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)



class Echo_CSS(torchvision.datasets.VisionDataset):
    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=40, period=3,
                 max_length=250,
                 clips=1,
                 pad=None,
                 ssl_type = 0,
                 ssl_postfix = "",
                 ssl_mult = 1,
                 ssl_edesonly = True,
                 noise=None,
                 target_transform=None,
                 external_test_location=None,
                 mask_gen = False,
                 mask_prop_range=(0.25, 0.5),
                 mask_n_boxes=3,
                 mask_random_aspect_ratio=True,
                 mask_prop_by_area=True,
                 mask_within_bounds=True,
                 mask_invert=True,
                 segin_dir = None
                 ):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.ssl_type = ssl_type
        self.ssl_postfix = ssl_postfix
        self.ssl_mult = ssl_mult

        self.ssl_edesonly = ssl_edesonly

        self.mask_gen = mask_gen
        self.prop_range = mask_prop_range
        self.n_boxes = mask_n_boxes
        self.random_aspect_ratio = mask_random_aspect_ratio
        self.prop_by_area = mask_prop_by_area
        self.within_bounds = mask_within_bounds
        self.invert = mask_invert

        self.frame_min = length * period

        self.segin_dir = segin_dir

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            print("Using data file from ", os.path.join(self.root, "FileList{}.csv".format(self.ssl_postfix)))

            with open(os.path.join(self.root, "FileList{}.csv".format(self.ssl_postfix))) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            if self.frame_min == 0:
                assert 1==2, "frame_min must > 0"
            else:
                data = data[data["NumberOfFrames"] > self.frame_min]
                print("total length after dropping short videos {}".format(len(data)), flush=True)

            if self.ssl_type == 1:
                assert self.split == "TRAIN", "subset selection only for train"
                #### labeled training
                data = data[data["SSL_SPLIT"] == "LABELED"]
                print("Using SSL_SPLIT Labeled, total samples", len(data))
                #### need to double/multiply the dataset 
                data_columns = data.columns
                data = pandas.DataFrame(np.repeat(data.values,self.ssl_mult,axis=0))
                data.columns = data_columns
                print("data after duplicates:", len(data))

            elif self.ssl_type == 2:
                assert self.split == "TRAIN", "subset selection only for train"
                ### unlableled training
                data = data[data["SSL_SPLIT"] == "UNLABELED"]
                print("Using SSL_SPLIT unlabeled, total samples", len(data))
                data_columns = data.columns
                data = pandas.DataFrame(np.repeat(data.values,self.ssl_mult,axis=0))
                data.columns = data_columns
                print("data after duplicates:", len(data))

            elif self.ssl_type == 0:
                print("Using SSL_SPLIT ALL, total samples", len(data))
                pass
            else:
                assert 1==2, "invalid option for ssl_type in echonet data"

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()

            self.outcome = data.values.tolist()


            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)


            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    filename = filename + ".avi"
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        if self.split == "EXTERNAL_TEST":
            video_path = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video_path = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video_path = os.path.join(self.root, "Videos", self.fnames[index])

        video = echonet.utils.loadvideo(video_path).astype(np.float32)

        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        f_old = f

        if f < length * self.period:
            assert 1 == 2, "something wrong with logic, frames should not be less then length * period = 80"
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633


        if self.pad is not None:
            i1_iekd, j1_iekd = np.random.randint(0, 2 * self.pad, 2)
        else:
            i1_iekd = 0
            j1_iekd = 0

        key = self.fnames[index]
        late_lb_frm = max([np.int(self.frames[key][-1]), np.int(self.frames[key][0])])
        erly_lb_frm = min([np.int(self.frames[key][-1]), np.int(self.frames[key][0])])
        if self.ssl_type == 1 or self.ssl_type == 0: 
            include_frm_rng = np.arange(max(late_lb_frm % self.period, late_lb_frm - (length-1)*self.period), min(f - (length - 1) * self.period, erly_lb_frm + self.period), self.period)
            start = np.random.choice(include_frm_rng, self.clips)
        else:
            include_frm_rng = np.arange(0, f - (length - 1) * self.period)

        if self.clips == "all":
            start = np.arange(f - (length - 1) * self.period)
        else:
            start = np.random.choice(include_frm_rng, self.clips)



        # Gather targets
        target = []
        target_iekd = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                if self.ssl_edesonly:
                    target.append(video[:, self.frames[key][-1], :, :])
                else:
                    frm_select = np.random.choice(video.shape[1])
                    target.append(video[:, frm_select, :, :])
            elif t == "SmallFrame":
                if self.ssl_edesonly:
                    target.append(video[:, self.frames[key][0], :, :])
                else:
                    frm_select = np.random.choice(video.shape[1])
                    target.append(video[:, frm_select, :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]

                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)

            elif t == "CYCLE":
                if self.ssl_type == 1 or self.ssl_type == 0 or True:
                    ## LARGE
                    include_frmlb_raw =  np.int(late_lb_frm)
                    include_frmlb2_raw = np.random.choice(np.arange(start, min(f_old, start + (length - 1) * self.period), self.period))

                    include_frmlb_num = np.int((include_frmlb_raw - start) / self.period)
                    include_frmlb2_num = np.int((include_frmlb2_raw - start) / self.period)
                    
                    target_iekd.append(include_frmlb_num)
                    target_iekd.append(include_frmlb2_num)
                    
                    if self.pad is not None:
                        vidframe_lb_tmp = np.zeros((3, video.shape[2] + 2 * self.pad, video.shape[3] + 2 * self.pad), dtype=video.dtype)
                        vidframe_lb_tmp[:, self.pad:-self.pad, self.pad:-self.pad] = video[:, include_frmlb_raw, :, :]
                        vidframe_lb = vidframe_lb_tmp[:, i1_iekd:(i1_iekd + h), j1_iekd:(j1_iekd + w)]

                        vidframe_lb2_tmp = np.zeros((3, video.shape[2] + 2 * self.pad, video.shape[3] + 2 * self.pad), dtype=video.dtype)
                        vidframe_lb2_tmp[:, self.pad:-self.pad, self.pad:-self.pad] = video[:, include_frmlb2_raw, :, :]
                        vidframe_lb2 = vidframe_lb2_tmp[:, i1_iekd:(i1_iekd + h), j1_iekd:(j1_iekd + w)]
                    else:
                        vidframe_lb = video[:, include_frmlb_raw, :, :]
                        vidframe_lb2 = video[:, include_frmlb2_raw, :, :]

                    target_iekd.append(vidframe_lb)
                    target_iekd.append(vidframe_lb2)

                    t = self.trace[key][late_lb_frm]
                    x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                    x = np.concatenate((x1[1:], np.flip(x2[1:])))
                    y = np.concatenate((y1[1:], np.flip(y2[1:])))

                    r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                    mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                    mask[r, c] = 1

                    if self.pad is not None:
                        mask_iekd = np.zeros((video.shape[2] + 2 * self.pad, video.shape[3] + 2 * self.pad), dtype=mask.dtype)
                        mask_iekd[self.pad:-self.pad, self.pad:-self.pad] = mask
                        mask = mask_iekd[i1_iekd:(i1_iekd + h), j1_iekd:(j1_iekd + w)]

                    target_iekd.append(mask)


                    t = self.trace[key][erly_lb_frm]
                    x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                    x = np.concatenate((x1[1:], np.flip(x2[1:])))
                    y = np.concatenate((y1[1:], np.flip(y2[1:])))

                    r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                    mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                    mask[r, c] = 1

                    if self.pad is not None:
                        mask_iekd = np.zeros((video.shape[2] + 2 * self.pad, video.shape[3] + 2 * self.pad), dtype=mask.dtype)
                        mask_iekd[self.pad:-self.pad, self.pad:-self.pad] = mask
                        mask = mask_iekd[i1_iekd:(i1_iekd + h), j1_iekd:(j1_iekd + w)]

                    target_iekd.append(mask)
                
                else:
                    target_iekd.append(None)
                    target_iekd.append(None)
                    target_iekd.append(None)
                    target_iekd.append(None)
                    target_iekd.append(None)
                    target_iekd.append(None)
                
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if self.mask_gen:
            assert self.pad is None, "cps mask generation does not consider padding augmentation atm "
            n_masks = 2
            mask_shape = (112, 112)

            rng = np.random

            if self.prop_by_area:
                # Choose the proportion of each mask that should be above the threshold
                mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

                # Zeros will cause NaNs, so detect and suppres them
                zero_mask = mask_props == 0.0

                if self.random_aspect_ratio:
                    y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                    x_props = mask_props / y_props
                else:
                    y_props = x_props = np.sqrt(mask_props)
                fac = np.sqrt(1.0 / self.n_boxes)
                y_props *= fac
                x_props *= fac

                y_props[zero_mask] = 0
                x_props[zero_mask] = 0
            else:
                if self.random_aspect_ratio:
                    y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                    x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                else:
                    x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                fac = np.sqrt(1.0 / self.n_boxes)
                y_props *= fac
                x_props *= fac


            sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])  
            if self.within_bounds:
                positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
                rectangles = np.append(positions, positions + sizes, axis=2)
                
            else:
                centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
                rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

            if self.invert:
                masks = np.zeros((n_masks, 1) + mask_shape) 
            else:
                masks = np.ones((n_masks, 1) + mask_shape)
            for i, sample_rectangles in enumerate(rectangles):
                for y0, x0, y1, x1 in sample_rectangles:
                    masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
            target.append(masks)

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        if target_iekd != []:
            target_iekd = tuple(target_iekd) if len(target_iekd) > 1 else target_iekd[0]
            if self.target_transform is not None:
                target_iekd = self.target_transform(target_iekd)

        # Select clips from video
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        
        if self.segin_dir:
            seg_infer_path = os.path.join(self.segin_dir, self.fnames[index].replace(".avi", ".npy"))
            seg_infer_logits = np.load(seg_infer_path)
            seg_infer_probs = expit(seg_infer_logits)
            seg_infer_prob_norm = seg_infer_probs * 2 - 1

            #### check if need to append 
            if f_old < length * self.period:
                seg_infer_prob_norm = np.concatenate((seg_infer_prob_norm, np.ones((length * self.period - f_old, h, w), video[0].dtype) * -1), axis=0)

            seg_infer_prob_norm = np.expand_dims(seg_infer_prob_norm, axis=0)

            seg_infer_prob_norm_samp =  tuple(seg_infer_prob_norm[:, s + self.period * np.arange(length), :, :] for s in start)
            


        if self.clips == 1:
            video = video[0]
            if self.segin_dir:
                seg_infer_prob_norm_samp = seg_infer_prob_norm_samp[0]
                video = np.concatenate((video, seg_infer_prob_norm_samp), axis=0)
        else:
            video = np.stack(video)
            if self.segin_dir:
                seg_infer_prob_norm_samp = np.stack(seg_infer_prob_norm_samp)
                video = np.concatenate((video, seg_infer_prob_norm_samp), axis=1)

        if self.target_type != ["EF"]:
            pass

        if self.pad is not None:
            jit1 = np.random.random()*0.1
            video1 = video.copy()


            c, l, h, w = video.shape

            temp1 = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp1[:, :, self.pad:-self.pad, self.pad:-self.pad] = video1

            i1 = i1_iekd
            j1 = j1_iekd

            video1 = temp1[:, :, i1:(i1 + h), j1:(j1 + w)]

        else:
            video1 = video.copy()
            i1 = 0
            j1 = 0
        if self.target_type == ["EF"]:
            return video1, target, target_iekd, start, video_path, i1, j1
        else:
            return video1, target, target_iekd, start, video_path, i1, j1

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)




class Echo_tskd(torchvision.datasets.VisionDataset):

    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 ssl_type = 0,
                 ssl_postfix = "",
                 ssl_mult = 1,
                 ssl_edesonly = True,
                 noise=None,
                 target_transform=None,
                 external_test_location=None,
                 mask_gen = False,
                 mask_prop_range=(0.25, 0.5),
                 mask_n_boxes=3,
                 mask_random_aspect_ratio=True,
                 mask_prop_by_area=True,
                 mask_within_bounds=True,
                 mask_invert=True,
                 segin_dir = None
                 ):
        if root is None:
            root = echonet.config.DATA_DIR

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.ssl_type = ssl_type
        self.ssl_postfix = ssl_postfix
        self.ssl_mult = ssl_mult

        self.ssl_edesonly = ssl_edesonly

        self.mask_gen = mask_gen
        self.prop_range = mask_prop_range
        self.n_boxes = mask_n_boxes
        self.random_aspect_ratio = mask_random_aspect_ratio
        self.prop_by_area = mask_prop_by_area
        self.within_bounds = mask_within_bounds
        self.invert = mask_invert

        self.segin_dir = segin_dir

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            print("Using data file from ", os.path.join(self.root, "FileList{}.csv".format(self.ssl_postfix)))

            with open(os.path.join(self.root, "FileList{}.csv".format(self.ssl_postfix))) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]


            if self.ssl_type == 1:
                assert self.split == "TRAIN", "subset selection only for train"
                #### labeled training
                data = data[data["SSL_SPLIT"] == "LABELED"]
                print("Using SSL_SPLIT Labeled, total samples", len(data))
                #### need to double/multiply the dataset 
                data_columns = data.columns
                data = pandas.DataFrame(np.repeat(data.values,self.ssl_mult,axis=0))
                data.columns = data_columns
                print("data after duplicates:", len(data))

            elif self.ssl_type == 2:
                assert self.split == "TRAIN", "subset selection only for train"
                ### unlableled training
                data = data[data["SSL_SPLIT"] == "UNLABELED"]
                print("Using SSL_SPLIT unlabeled, total samples", len(data))
                data_columns = data.columns
                data = pandas.DataFrame(np.repeat(data.values,self.ssl_mult,axis=0))
                data.columns = data_columns
                print("data after duplicates:", len(data))

            elif self.ssl_type == 0:
                print("Using SSL_SPLIT ALL, total samples", len(data))
                pass
            else:
                assert 1==2, "invalid option for ssl_type in echonet data"

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()

            self.outcome = data.values.tolist()


            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            # print(self.fnames)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    filename = filename + ".avi"
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
        # print(self.fnames)

    def __getitem__(self, index):
        # Find filename of video

        if self.split == "EXTERNAL_TEST":
            video_path = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video_path = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video_path = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video_path).astype(np.float32)

        if self.noise is not None:
            n = video.shape[1] * video.shape[2] * video.shape[3]
            ind = np.random.choice(n, round(self.noise * n), replace=False)
            f = ind % video.shape[1]
            ind //= video.shape[1]
            i = ind % video.shape[2]
            ind //= video.shape[2]
            j = ind
            video[:, f, i, j] = 0

        # Apply normalization
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        f_old = f

        if f < length * self.period:
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        ### Determine if we want to include a large frame or a small frame
        lrgsml_rnd = np.random.choice(np.array([0,1]))
        key = self.fnames[index]
        if lrgsml_rnd == 0:
            include_frm = np.int(self.frames[key][-1])
        else:
            include_frm = np.int(self.frames[key][0])

        if self.pad is not None:
            i1_iekd, j1_iekd = np.random.randint(0, 2 * self.pad, 2)
        else:
            i1_iekd = 0
            j1_iekd = 0

        include_frm_rng = np.arange(max(include_frm % self.period, include_frm - (length-1)*self.period), min(f - (length - 1) * self.period, include_frm + self.period), self.period)
        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            start = np.random.choice(include_frm_rng, self.clips)

        # Gather targets
        target = []
        target_iekd = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                target.append(np.int(self.frames[key][-1]))
            elif t == "SmallIndex":
                target.append(np.int(self.frames[key][0]))
            elif t == "LargeFrame":
                if self.ssl_edesonly:
                    target.append(video[:, self.frames[key][-1], :, :])
                else:
                    frm_select = np.random.choice(video.shape[1])
                    target.append(video[:, frm_select, :, :])
            elif t == "SmallFrame":
                if self.ssl_edesonly:
                    target.append(video[:, self.frames[key][0], :, :])
                else:
                    frm_select = np.random.choice(video.shape[1])
                    target.append(video[:, frm_select, :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]

                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)

            elif t == "IEKD":
                if lrgsml_rnd == 0:
                    ## LARGE
                    include_frmlb_raw =  np.int(self.frames[key][-1])
                    include_frmunlb_raw = np.random.choice(np.arange(start, min(f_old, start + (length - 1) * self.period), self.period))

                    include_frmlb_num = np.int((include_frmlb_raw - start) / self.period)
                    include_frmunlb_num = np.int((include_frmunlb_raw - start) / self.period)
                    target_iekd.append(include_frmlb_num)
                    target_iekd.append(include_frmunlb_num)
                    
                    if self.pad is not None:
                        vidframe_lb_tmp = np.zeros((3, video.shape[2] + 2 * self.pad, video.shape[3] + 2 * self.pad), dtype=video.dtype)
                        vidframe_lb_tmp[:, self.pad:-self.pad, self.pad:-self.pad] = video[:, include_frmlb_raw, :, :]
                        vidframe_lb = vidframe_lb_tmp[:, i1_iekd:(i1_iekd + h), j1_iekd:(j1_iekd + w)]

                        vidframe_unlb_tmp = np.zeros((3, video.shape[2] + 2 * self.pad, video.shape[3] + 2 * self.pad), dtype=video.dtype)
                        vidframe_unlb_tmp[:, self.pad:-self.pad, self.pad:-self.pad] = video[:, include_frmunlb_raw, :, :]
                        vidframe_unlb = vidframe_unlb_tmp[:, i1_iekd:(i1_iekd + h), j1_iekd:(j1_iekd + w)]
                    else:
                        vidframe_lb = video[:, include_frmlb_raw, :, :]
                        vidframe_unlb = video[:, include_frmunlb_raw, :, :]

                    target_iekd.append(vidframe_lb)
                    target_iekd.append(vidframe_unlb)

                    t = self.trace[key][self.frames[key][-1]]
                    x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                    x = np.concatenate((x1[1:], np.flip(x2[1:])))
                    y = np.concatenate((y1[1:], np.flip(y2[1:])))

                    r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                    mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                    mask[r, c] = 1

                    if self.pad is not None:
                        mask_iekd = np.zeros((video.shape[2] + 2 * self.pad, video.shape[3] + 2 * self.pad), dtype=mask.dtype)
                        mask_iekd[self.pad:-self.pad, self.pad:-self.pad] = mask
                        mask = mask_iekd[i1_iekd:(i1_iekd + h), j1_iekd:(j1_iekd + w)]

                    target_iekd.append(mask)
                else:

                    include_frmlb_raw = np.int(self.frames[key][0])
                    include_frmunlb_raw = np.random.choice(np.arange(start, min(f_old, start + (length - 1) * self.period), self.period))

                    include_frmlb_num = np.int((include_frmlb_raw - start) / self.period)
                    include_frmunlb_num = np.int((include_frmunlb_raw - start) / self.period)

                    target_iekd.append(include_frmlb_num)
                    target_iekd.append(include_frmunlb_num) ### avoids 0 pad frames

                    if self.pad is not None:
                        vidframe_lb_tmp = np.zeros((3, video.shape[2] + 2 * self.pad, video.shape[3] + 2 * self.pad), dtype=video.dtype)
                        vidframe_lb_tmp[:, self.pad:-self.pad, self.pad:-self.pad] = video[:, include_frmlb_raw, :, :]
                        vidframe_lb = vidframe_lb_tmp[:, i1_iekd:(i1_iekd + h), j1_iekd:(j1_iekd + w)]

                        vidframe_unlb_tmp = np.zeros((3, video.shape[2] + 2 * self.pad, video.shape[3] + 2 * self.pad), dtype=video.dtype)
                        vidframe_unlb_tmp[:, self.pad:-self.pad, self.pad:-self.pad] = video[:, include_frmunlb_raw, :, :]
                        vidframe_unlb = vidframe_unlb_tmp[:, i1_iekd:(i1_iekd + h), j1_iekd:(j1_iekd + w)]
                    else:
                        vidframe_lb = video[:, include_frmlb_raw, :, :]
                        vidframe_unlb = video[:, include_frmunlb_raw, :, :]

                    target_iekd.append(vidframe_lb)
                    target_iekd.append(vidframe_unlb)

                    t = self.trace[key][self.frames[key][0]]
                    x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                    x = np.concatenate((x1[1:], np.flip(x2[1:])))
                    y = np.concatenate((y1[1:], np.flip(y2[1:])))

                    r, c = skimage.draw.polygon(np.rint(y).astype(np.int), np.rint(x).astype(np.int), (video.shape[2], video.shape[3]))
                    mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                    mask[r, c] = 1

                    if self.pad is not None:
                        mask_iekd = np.zeros((video.shape[2] + 2 * self.pad, video.shape[3] + 2 * self.pad), dtype=mask.dtype)
                        mask_iekd[self.pad:-self.pad, self.pad:-self.pad] = mask
                        mask = mask_iekd[i1_iekd:(i1_iekd + h), j1_iekd:(j1_iekd + w)]

                    target_iekd.append(mask)

            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if self.mask_gen:
            n_masks = 2
            mask_shape = (112, 112)

            rng = np.random

            if self.prop_by_area:
                # Choose the proportion of each mask that should be above the threshold
                mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

                # Zeros will cause NaNs, so detect and suppres them
                zero_mask = mask_props == 0.0

                if self.random_aspect_ratio:
                    y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                    x_props = mask_props / y_props
                else:
                    y_props = x_props = np.sqrt(mask_props)
                fac = np.sqrt(1.0 / self.n_boxes)
                y_props *= fac
                x_props *= fac

                y_props[zero_mask] = 0
                x_props[zero_mask] = 0
            else:
                if self.random_aspect_ratio:
                    y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                    x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                else:
                    x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                fac = np.sqrt(1.0 / self.n_boxes)
                y_props *= fac
                x_props *= fac


            sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])  

            if self.within_bounds:
                positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
                rectangles = np.append(positions, positions + sizes, axis=2)
                
            else:
                centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
                rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

            if self.invert:
                masks = np.zeros((n_masks, 1) + mask_shape)  ### gives a zeros matrix of shape (2,1,112,112)
            else:
                masks = np.ones((n_masks, 1) + mask_shape)
            for i, sample_rectangles in enumerate(rectangles):
                for y0, x0, y1, x1 in sample_rectangles:
                    masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
            target.append(masks)

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        if target_iekd != []:
            target_iekd = tuple(target_iekd) if len(target_iekd) > 1 else target_iekd[0]
            if self.target_transform is not None:
                target_iekd = self.target_transform(target_iekd)

        # Select clips from video
        video = tuple(video[:, s + self.period * np.arange(length), :, :] for s in start)
        
        if self.segin_dir:
            seg_infer_path = os.path.join(self.segin_dir, self.fnames[index].replace(".avi", ".npy"))
            seg_infer_logits = np.load(seg_infer_path)
            seg_infer_probs = expit(seg_infer_logits)
            seg_infer_prob_norm = seg_infer_probs * 2 - 1

            #### check if need to append 
            if f_old < length * self.period:
                seg_infer_prob_norm = np.concatenate((seg_infer_prob_norm, np.ones((length * self.period - f_old, h, w), video[0].dtype) * -1), axis=0)

            seg_infer_prob_norm = np.expand_dims(seg_infer_prob_norm, axis=0)

            seg_infer_prob_norm_samp =  tuple(seg_infer_prob_norm[:, s + self.period * np.arange(length), :, :] for s in start)
            


        if self.clips == 1:
            video = video[0]
            if self.segin_dir:
                seg_infer_prob_norm_samp = seg_infer_prob_norm_samp[0]
                video = np.concatenate((video, seg_infer_prob_norm_samp), axis=0)
        else:
            video = np.stack(video)
            if self.segin_dir:
                seg_infer_prob_norm_samp = np.stack(seg_infer_prob_norm_samp)
                video = np.concatenate((video, seg_infer_prob_norm_samp), axis=1)

        if self.target_type != ["EF"]:
            pass

        if self.pad is not None:
            video1 = video.copy()


            c, l, h, w = video.shape

            temp1 = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp1[:, :, self.pad:-self.pad, self.pad:-self.pad] = video1

            i1 = i1_iekd
            j1 = j1_iekd

            video1 = temp1[:, :, i1:(i1 + h), j1:(j1 + w)]


        else:
            video1 = video.copy()
            i1 = 0
            j1 = 0
        if self.target_type == ["EF"]:
            return video1, target, target_iekd, start, video_path, i1, j1
        else:
            # return video1, target
            return video1, target, target_iekd, start, video_path, i1, j1

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)






def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)





