import os
import re
import glob
import argparse
import pandas as pd
from tqdm import tqdm


POSCONS = [         "JCP2022_037716",
                    "JCP2022_064022",
                    "JCP2022_025848",
                    "JCP2022_050797",
                    "JCP2022_046054",
                    "JCP2022_012818",
                    "JCP2022_085227",
                    "JCP2022_035095",
                    ]

FILT_BATCHES_S3 = ['CP_33_all_Phenix1', 'CP_36_all_Phenix1', 'CP_35_all_Phenix1', 'CP_34_mix_Phenix1']

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_root = os.path.join(repo_root, "data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        default=os.path.join(data_root, "all_indices"),
        help="Path to parquet index file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(data_root, "indices"),
        help="Path to parquet index file",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default=os.path.join(data_root, "jumpcp-subset"),
        help="Path to parquet index file",
    )
    parser.add_argument(
        "--metadatadir",
        type=str,
        default=os.path.join(data_root, "metadata"),
        help="Path to parquet index file",
    )
    parser.add_argument(
        "--num-min-samples",
        type=int,
        default=36,
        help="Minimum number of samples per plate position",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default=os.path.join(data_root, "jumpcp"),
        help="Path to parquet index file",
    )

    args = parser.parse_args()
    args.indir = os.path.abspath(os.path.expanduser(args.indir))
    args.outdir = os.path.abspath(os.path.expanduser(args.outdir))
    args.datadir = os.path.abspath(os.path.expanduser(args.datadir))
    args.metadatadir = os.path.abspath(os.path.expanduser(args.metadatadir))

    return args


if __name__ == "__main__":
    args = parse_args()

    dirs = sorted(os.listdir(args.indir))

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    well = pd.read_csv(os.path.join(args.metadatadir, "well.csv"))
    plate = pd.read_csv(os.path.join(args.metadatadir, "plate.csv"))

    all_controls, all_poscons = [], []

    for d in dirs:
        folder = os.path.join(args.indir, d)
        regex = f"{folder}/*/*/*"

        indices = glob.glob(regex)

        source_index = pd.concat([pd.read_csv(f) for f in indices], join="inner")

        assert len(source_index["Metadata_Source"].unique()) == 1, "Images from several sources are bein mixed in the same index."
        current_source = source_index["Metadata_Source"].unique()[0]

        filename, _ = os.path.splitext(d)

        channels = source_index.filter(regex='PathName_Orig*', axis='columns').columns

        for c in tqdm(channels):
            match = re.search('PathName_Orig(?P<component>.*)', c)
            component = match['component']

            # get path to download files in AWS
            aws_path = source_index.apply(lambda df: os.path.join(df[c], df[f"FileName_Orig{component}"]), axis=1)
            aws_path = aws_path.str.replace(f's3://cellpainting-gallery/', '')
            source_index[f"AWS_Path{component}"] = aws_path


            source_index[c] = source_index[c].str.replace(f's3://cellpainting-gallery/cpg0016-jump/{d}/images', os.path.join(args.datadir, f"{d}"))
            source_index[component] = source_index.apply(lambda df: os.path.join(df[c], df[f"FileName_Orig{component}"]), axis=1)

        source_index['Metadata_Plate'] = source_index['Metadata_Plate'].astype(str)

        source_index = source_index.merge(well, 
                            on=[
                                "Metadata_Source",
                                "Metadata_Plate",
                                "Metadata_Well",
                            ],
                            how="left"
                            ) 

        source_index.dropna(subset="Metadata_JCP2022", inplace=True)


        source_index['Metadata_Plate_ID'] = source_index.apply(lambda x: f"{x['Metadata_Source']}_{x['Metadata_Batch']}_{x['Metadata_Plate']}", axis=1)
        source_index['Metadata_Well_ID'] = source_index.apply(lambda x: f"{x['Metadata_Source']}_{x['Metadata_Batch']}_{x['Metadata_Plate']}_{x['Metadata_Well']}", axis=1)
        source_index['Metadata_Sample_ID'] = source_index.apply(lambda x: f"{x['Metadata_Source']}_{x['Metadata_Batch']}_{x['Metadata_Plate']}_{x['Metadata_Well']}_{x['Metadata_Site']}", axis=1)

        source_plates = plate[plate["Metadata_Source"] == current_source]
        compound_plates = source_plates[source_plates["Metadata_PlateType"] == "COMPOUND"]["Metadata_Plate"].tolist()
        source_index = source_index[source_index["Metadata_Plate"].isin(compound_plates)]

        source_index["img_path"] = args.img_path

        if current_source == "source_3":
            source_index = source_index[~source_index["Metadata_Batch"].isin(FILT_BATCHES_S3)]

        # these lines are just to have the same order in the index file as the one used to run the experiments
        # because the result of glob.glob might vary accross machines, but the samples used should be the same
        server_index_file = os.path.join(args.outdir, f"{filename}_server.pq")

        source_controls = source_index[source_index['Metadata_JCP2022'] == "JCP2022_033924"]
        source_poscons = source_index[source_index['Metadata_JCP2022'].isin(POSCONS)]

        pos_to_filter = source_poscons.groupby(["Metadata_JCP2022", "Metadata_Well"]).size()[source_poscons.groupby(["Metadata_JCP2022", "Metadata_Well"]).size() <= args.num_min_samples].index

        for mol, pos in pos_to_filter:
            source_poscons = source_poscons[~((source_poscons["Metadata_JCP2022"] == mol) & (source_poscons["Metadata_Well"] == pos))]

        if os.path.isfile(server_index_file):
            server_index = pd.read_parquet(server_index_file)
            server_index.set_index("Metadata_Sample_ID", inplace=True)
            source_poscons.set_index("Metadata_Sample_ID", inplace=True)
            source_poscons = source_poscons.loc[server_index.index]
            source_poscons = source_poscons.reset_index()

        source_controls = source_controls.astype(str)
        source_poscons = source_poscons.astype(str)

        source_controls.to_parquet(os.path.join(args.outdir, f"{filename}_controls.pq"))
        source_poscons.to_parquet(os.path.join(args.outdir, f"{filename}_poscons.pq"))

        all_controls.append(source_controls)
        all_poscons.append(source_poscons)

    all_controls = pd.concat(all_controls)
    all_poscons = pd.concat(all_poscons)

    all_poscons["Metadata_Row"] = all_poscons["Metadata_Row"].astype(str)
    all_controls["Metadata_Row"] = all_controls["Metadata_Row"].astype(str)

    all_poscons.to_parquet(os.path.join(args.outdir, f"all_poscons.pq"))
    all_controls.to_parquet(os.path.join(args.outdir, f"all_controls.pq"))