import random

import cv2
import numpy as np
import torch
from skimage import io


s1_min = np.array([-25 , -62 , -25, -60], dtype="float32")
s1_max = np.array([ 29 ,  28,  30,  22 ], dtype="float32")
s1_mm = s1_max - s1_min

s2_max = np.array(
    [19616., 18400., 17536., 17097., 16928., 16768., 16593., 16492., 15401., 15226.,   255.],
    dtype="float32",
)

IMG_SIZE = (256, 256)


def read_imgs(chip_id, data_dir):
    imgs, mask = [], []
    for month in range(12):
        img_s1 = io.imread(data_dir / f"{chip_id}_S1_{month:0>2}.tif")
        m = img_s1 == -9999
        img_s1 = img_s1.astype("float32")
        img_s1 = (img_s1 - s1_min) / s1_mm
        img_s1 = np.where(m, 0, img_s1)
        filepath = data_dir / f"{chip_id}_S2_{month:0>2}.tif"
        if filepath.is_file():
            img_s2 = io.imread(filepath)
            img_s2 = img_s2.astype("float32")
            img_s2 = img_s2 / s2_max
        else:
            img_s2 = np.zeros(IMG_SIZE + (11,), dtype="float32")

        img = np.concatenate([img_s1, img_s2], axis=2)
        img = np.transpose(img, (2, 0, 1))
        imgs.append(img)
        mask.append(False)

    mask = np.array(mask)

    imgs = np.stack(imgs, axis=0)  # [t, c, h, w]

    return imgs, mask


def rotate_image(image, angle, rot_pnt, scale=1):
    rot_mat = cv2.getRotationMatrix2D(rot_pnt, angle, scale)
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) #INTER_NEAREST

    return result


def train_aug(imgs, mask, target):
    # imgs: [t, c, h, w]
    # mask: [t, ]
    # target: [h, w]
    if random.random() > 0.5:  # horizontal flip
        imgs = imgs[..., ::-1]
        target = target[..., ::-1]

    k = random.randrange(4)
    if k > 0:  # rotate90
        imgs = np.rot90(imgs, k=k, axes=(-2, -1))
        target = np.rot90(target, k=k, axes=(-2, -1))

    if random.random() > 0.3:  # scale-rotate
        _d = int(imgs.shape[2] * 0.1)  # 0.4)
        rot_pnt = (imgs.shape[2] // 2 + random.randint(-_d, _d), imgs.shape[3] // 2 + random.randint(-_d, _d))
        #scale = 1
        #if random.random() > 0.2:
            #scale = random.normalvariate(1.0, 0.1)

        #angle = 0
        #if random.random() > 0.2:
        angle = random.randint(0, 90) - 45

        if (angle != 0):# or (scale != 1):
            t = len(imgs)  # t, c, h, w
            imgs = np.concatenate(imgs, axis=0)  # t*c, h, w
            imgs = np.transpose(imgs, (1, 2, 0))  # h, w, t*c
            imgs = rotate_image(imgs, angle, rot_pnt)
            imgs = np.transpose(imgs, (2, 0, 1))  # t*c, h, w
            imgs = np.reshape(imgs, (t, -1, imgs.shape[1], imgs.shape[2]))  # t, c, h, w
            target = rotate_image(target, angle, rot_pnt)

    if random.random() > 0.5:  # "word" dropout
        while True:
            mask2 = np.random.rand(*mask.shape) < 0.3
            mask3 = np.logical_or(mask, mask2)
            if not mask3.all():
                break

        mask = mask3
        imgs[mask2] = 0

    return imgs.copy(), mask, target.copy()


class DS(torch.utils.data.Dataset):
    def __init__(self, df, dir_features, dir_labels=None, augs=False):
        self.df = df
        self.dir_features = dir_features
        self.dir_labels = dir_labels
        self.augs = augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]

        imgs, mask = read_imgs(item.chip_id, self.dir_features)
        if self.dir_labels is not None:
            target = io.imread(self.dir_labels / f'{item.chip_id}_agbm.tif')
        else:
            target = item.chip_id

        if self.augs:
            imgs, mask, target = train_aug(imgs, mask, target)

        return imgs, mask, target


def predict_tta(models, images, masks, ntta=1):
    result = images.new_zeros((images.shape[0], 1, images.shape[-2], images.shape[-1]))
    n = 0
    for model in models:
        logits = model(images, masks)
        result += logits
        n += 1

        if ntta == 2:
            # hflip
            logits = model(torch.flip(images, dims=[-1]), masks)
            result += torch.flip(logits, dims=[-1])
            n += 1

        if ntta == 3:
            # vflip
            logits = model(torch.flip(images, dims=[-2]), masks)
            result += torch.flip(logits, dims=[-2])
            n += 1

        if ntta == 4:
            # hvflip
            logits = model(torch.flip(images, dims=[-2, -1]), masks)
            result += torch.flip(logits, dims=[-2, -1])
            n += 1

    result /= n * len(models)

    return result
