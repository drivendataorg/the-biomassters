import argparse
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import pandas as pd
import torch
import torch.distributed
import torch.utils
import torch.utils.data
import tqdm
import PIL.Image as Image

import dataset


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-df",
        type=str,
        default="./data/features_metadata.csv",
        help="path to test df",
    )
    parser.add_argument(
        "--test-images-dir",
        type=str,
        help="path to test dir",
        default="./data/test_features",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="path models",
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="output directory",
        required=True,
    )

    parser.add_argument(
        "--num-workers", type=int, help="number of data loader workers", default=8,
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=32)

    parser.add_argument(
        "--tta",
        type=int,
        help="tta",
        default=1,
    )
    parser.add_argument("--img-size", type=int, nargs=2, default=dataset.IMG_SIZE)

    args = parser.parse_args(args=args)

    return args


def main():
    args = parse_args()
    print(args)

    # torch.jit.enable_onednn_fusion(True)

    model = torch.load(args.model_path)
    model = model.eval()
    model = model.cuda()
    model = model.to(memory_format=torch.channels_last)
    models = [model]

    df = pd.read_csv(args.test_df)
    test_df = df[df.split == "test"].copy()
    test_df = test_df.groupby("chip_id").agg(list).reset_index()
    print(test_df)

    test_images_dir = Path(args.test_images_dir)
    test_dataset = dataset.DS(
        df=test_df,
        dir_features=test_images_dir,
    )
    test_sampler = None

    args.num_workers = min(args.batch_size, 4)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=False,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    with torch.no_grad():
        with tqdm.tqdm(test_loader, leave=False, mininterval=2) as pbar:
            for images, mask, target in pbar:
                images = images.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                logits = dataset.predict_tta(models, images, mask, ntta=args.tta)

                logits = logits.squeeze(1).cpu().numpy()

                for pred, chip_id in zip(logits, target):
                    im = Image.fromarray(pred)
                    im.save(out_dir / f"{chip_id}_agbm.tif", format="TIFF", save_all=True)

                torch.cuda.synchronize()


if __name__ == "__main__":
    main()
