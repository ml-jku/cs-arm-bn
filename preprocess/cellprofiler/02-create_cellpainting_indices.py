import os
import re
import glob
import torch 
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_root = os.path.join(repo_root, "data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to parquet index file",
    )
    parser.add_argument(
        "--indir",
        type=str,
        default=os.path.join(data_root, "cellprofiler_features", "aws"), 
        help="Path to parquet index file",
    )
    parser.add_argument(
        "--metadatadir",
        type=str,
        default=os.path.join(data_root, "metadata"),
        help="Path to parquet index file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(data_root, "cellprofiler_features"),  
        help="Path to parquet index file",
    )
    args = parser.parse_args()

    return args


def main(args):
    well_df = pd.read_csv(os.path.join(args.metadatadir, "well.csv"))

    images_df = pd.read_parquet(args.index)
    images_df['Metadata_Well_ID_CP'] = images_df.apply(lambda x: f"{x['Metadata_Source']}_{x['Metadata_Plate']}_{x['Metadata_Well']}", axis=1)

    all_files = glob.glob(f"{args.indir}/*/*/*/*.parquet")

    filelst = []

    for f in all_files:
        print(f)
        df = pd.read_parquet(f)
        filelst.append(df)
    
    final_df = pd.concat(filelst)

    final_df['Metadata_Well_ID'] = final_df.apply(lambda x: f"{x['Metadata_Source']}_{x['Metadata_Plate']}_{x['Metadata_Well']}", axis=1)
    filtered_cellpainting = final_df[final_df["Metadata_Well_ID"].isin(images_df["Metadata_Well_ID_CP"])]
    well_df['Metadata_Well_ID'] = well_df.apply(lambda x: f"{x['Metadata_Source']}_{x['Metadata_Plate']}_{x['Metadata_Well']}", axis=1)

    filtered_cellpainting.set_index("Metadata_Well_ID", inplace=True)
    well_df.set_index("Metadata_Well_ID", inplace=True)

    filtered_cellpainting_with_labels = filtered_cellpainting.merge(well_df["Metadata_JCP2022"], left_index=True, right_index=True, how="inner")
    filtered_cellpainting_metadata = filtered_cellpainting_with_labels.loc[:, ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]].reset_index()
    filtered_cellpainting_metadata = filtered_cellpainting_metadata.merge(images_df[["Metadata_Plate", "Metadata_Batch"]].drop_duplicates(), left_on="Metadata_Plate", right_on="Metadata_Plate", how="inner" )

    np_cellpainting_features = filtered_cellpainting_with_labels.select_dtypes(include="number").to_numpy()
    filtered_cellpainting_labels = filtered_cellpainting_with_labels["Metadata_JCP2022"].to_numpy()
    filtered_cellpainting_metadata = filtered_cellpainting_metadata.loc[:, ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]].reset_index(drop=True)

    # Filter columns with nans and zero variance within each source
    mask = ~np.isnan(np_cellpainting_features).any(axis=0)
    np_cellpainting_features = np_cellpainting_features[:, mask]
    np_cellpainting_features = torch.from_numpy(np_cellpainting_features)

    groups = filtered_cellpainting_metadata["Metadata_Source"].to_numpy()
    unique_groups = np.unique(groups)

    if len(unique_groups) > 1:
        group_vars = np.vstack([
            torch.var(np_cellpainting_features[groups == g], axis=0)
            for g in unique_groups
        ])
        
        mask = (group_vars != 0).all(axis=0)
        np_cellpainting_features = np_cellpainting_features[:, mask]
        

    basename, _ = os.path.splitext(os.path.basename(args.index))

    features_path = os.path.join(args.outdir, f"{basename}_features.npy")
    labels_path = os.path.join(args.outdir, f"{basename}_labels.npy")
    metadata_path = os.path.join(args.outdir, f"{basename}_metadata.pq")

    # SAVE
    np.save(features_path, np_cellpainting_features.numpy())
    np.save(labels_path, filtered_cellpainting_labels)
    filtered_cellpainting_metadata.to_parquet(metadata_path)


if __name__ == "__main__":
    args = parse_args()

    main(args)