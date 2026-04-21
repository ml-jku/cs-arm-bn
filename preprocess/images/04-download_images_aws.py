
import os
import argparse 
import pandas as pd
import numpy as np 
from io import BytesIO
from PIL import Image

import boto3
import pyvips
from tqdm import tqdm
from botocore import UNSIGNED
from botocore.config import Config

from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BUCKET_NAME = "cellpainting-gallery"
s3_client = None

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, required=True, help="Path to parquet index file")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(repo_root, "data", "jumpcp"),
        help="Directory where downloaded images will be written",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        choices=[3, 5],
        default=5,
        help="Number of channels expected in the input index",
    )
    args = parser.parse_args()
    args.filename = os.path.abspath(os.path.expanduser(args.filename))
    args.outdir = os.path.abspath(os.path.expanduser(args.outdir))
    return args


def initialize():
  global s3_client
  s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))


def robust_convert_to_8bit(img, percentile=1.0):
    """Convert a array to a 8-bit image by min percentile normalisation and clipping."""
    img = img.astype(np.float32)
    img = (img - np.percentile(img, percentile)) / (
        np.percentile(img, 100 - percentile)
        - np.percentile(img, percentile)
        + np.finfo(float).eps
    )
    img = np.clip(img, 0, 1)

    img = (img * 255).astype(np.uint8)
    return img



def illumination_threshold(arr, perc=0.01):
    """ Return threshold value to not display a percentage of highest pixels"""

    perc = perc/100

    h = arr.shape[0]
    w = arr.shape[1]

    # find n pixels to delete
    total_pixels = h * w
    n_pixels = total_pixels * perc
    n_pixels = int(np.around(n_pixels))

    # find indexes of highest pixels
    flat_inds = np.argpartition(arr, -n_pixels, axis=None)[-n_pixels:]
    inds = np.array(np.unravel_index(flat_inds, arr.shape)).T

    max_values = [arr[i, j] for i, j in inds]

    threshold = min(max_values)

    return threshold


def sixteen_to_eight_bit_with_illum(arr, display_max, display_min=0):
    threshold_image = ((arr.astype(float) - display_min) * (arr > display_min))

    scaled_image = (threshold_image * (255 / (display_max - display_min)))
    scaled_image[scaled_image > 255] = 255

    scaled_image = scaled_image.astype(np.uint8)

    return scaled_image


def get_filelist(file, download_dir, n_channels=5):
    if n_channels == 5:
        channels = ["RNA", "ER", "AGP", "Mito", "DNA"]
    elif n_channels == 3:
        channels = ["DNA", "ER", "AGP"]

    df = pd.read_parquet(file)

    filelist = [df[f"URL_Orig{c}"].str[len("s3://cellpainting-gallery/"):].tolist() for c in channels]
    sample_ids = df["Metadata_Sample_ID"].tolist()
    ext = "tiff" if n_channels == 5 else "png"
    download_paths = [os.path.join(download_dir, f"{s}.{ext}") for s in sample_ids]

    all_files = list(zip(*filelist, download_paths))

    return all_files


def download_object(filedata):
    """Downloads an object from S3 to local."""
    chfiles = filedata[:-1]
    download_path = filedata[-1]

    final_arr = []

    for f in chfiles:
        with BytesIO() as data:
            s3_client.download_fileobj(BUCKET_NAME, f, data)
            arr = pyvips.Image.new_from_buffer(data.getvalue(), "")
            arr = arr.resize(1/2, kernel="nearest").numpy()
            thres = illumination_threshold(arr)
            arr_8bit = sixteen_to_eight_bit_with_illum(arr, thres)
            final_arr.append(arr_8bit)
            

    final_arr = np.stack(final_arr, axis=2)
    n_channels = final_arr.shape[2]
    final_arr = pyvips.Image.new_from_array(final_arr)

    print(f"Downloading {download_path}")

    if n_channels == 5:
        final_arr.tiffsave(download_path, pyramid=False, compression="deflate")
    elif n_channels == 3:
        final_arr.pngsave(download_path)
    return "Success"


def download_parallel_multiprocessing(keys_to_download):
    with ProcessPoolExecutor(initializer=initialize) as executor:
        future_to_key = {executor.submit(download_object, key): key for key in keys_to_download}

        for future in futures.as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()

            if not exception:
                yield key, future.result()
            else:
                yield key, exception


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    filelist = get_filelist(args.filename, args.outdir, n_channels=args.n_channels)

    initialize()
    
    for key, result in download_parallel_multiprocessing(filelist):
        print(f"{key}: {result}")


if __name__ == "__main__":
    args = parse_args()

    main(args)
