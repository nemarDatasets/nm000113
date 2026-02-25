README_CONTENT = """**Introduction:**
The 2020 BCI Competition Track 3 dataset contains EEG recordings from participants performing imagined speech tasks. This dataset was designed for brain-computer interface research focused on decoding imagined speech from brain signals. The dataset includes recordings from 15 subjects performing five different imagined speech commands: "Hello", "Help me", "Stop", "Thank you", and "Yes". The data is divided into training, validation, and test sets to facilitate machine learning approaches to imagined speech classification.

**Overview of the experiment:**
Participants performed imagined speech tasks where they were instructed to mentally articulate five different phrases without producing any audible speech or overt mouth movements. The five imagined speech commands were: "Hello", "Help me", "Stop", "Thank you", and "Yes". EEG signals were recorded during these mental articulation tasks. The dataset is split into three sets: Training Set (run-00), Validation Set (run-01), and Test Set (run-02). Each recording session contains multiple trials of imagined speech, with each trial corresponding to one of the five command categories. The EEG data was recorded using a multi-channel EEG system, and the exact number of channels and their montage are preserved in the BIDS format.

**Description of the preprocessing if any:**
The original MATLAB (.mat) files from the BCI Competition have been converted to BIDS-compliant EDF format. For training and validation sets, the data was stored in structured MATLAB arrays with fields for EEG data ('x'), labels ('y'), sampling frequency ('fs'), and channel labels ('clab'). For the test set, the data was stored in HDF5 format and labels were extracted from the Track3_Answer Sheet_Test.xlsx file. The EEG data has been scaled from the original units to Volts (multiplied by 1e-6). The epoched data structure from the original dataset has been concatenated into continuous recordings for BIDS compliance, with annotations marking the onset and duration of each imagined speech trial. Channel names and montage information from the original 'mnt' (montage) structure have been preserved in the BIDS format.

**Description of the event values if any:**
The events.tsv files contain annotations for each imagined speech trial. Each event has:
- onset: Time in seconds from the beginning of the recording when the imagined speech trial begins
- duration: Duration of the trial in seconds (calculated as the number of samples in the epoch divided by the sampling frequency)
- value: The imagined speech command label, one of: "Hello", "Help me", "Stop", "Thank you", or "Yes"
- trial_type: Corresponds to the value field

These annotations enable temporal segmentation of the continuous EEG data by imagined speech command type. The labels for the training and validation sets were extracted from the 'y' field in the original MATLAB structures (one-hot encoded vectors converted to class indices). For the test set, labels were obtained from the Track3_Answer Sheet_Test.xlsx file provided with the competition data.

**Citation:**
When using this dataset, please cite:

1. The 2020 BCI Competition Track 3: https://osf.io/pq7vb/overview

2. Original competition organizers and data collectors (please refer to the competition website for complete citation information)

**Data curators:**
Pierre Guetschel (BIDS conversion)

Competition co-chairs: Seong-Whan Lee, Klaus-Robert Müller, José del R. Millán
"""

DATASET_NAME = "2020 BCI competition, track 3"

from pathlib import Path
import re
import shutil

import scipy
import h5py
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description, make_report
import mne
import numpy as np
import pandas as pd


SETS = {"Training": 0, "Validation": 1, "Test": 2}
EPO_SUFFIX = {"00": "train", "01": "validation", "02": "test"}
LABELS = ["Hello", "Help me", "Stop", "Thank you", "Yes"]


def _get_records(source_root: Path):
    track3_root = (
        source_root / "Pq7vb" / "osfstorage" / "Track#3 Imagined speech classification"
    )
    assert track3_root.exists(), f"Track 3 root {track3_root} does not exist"

    # example record:
    # "Training Set/Data_Sample01.mat"
    record_regex = r"Data_Sample(?P<subject>[0-9]+)\.mat"

    records = []
    for set_name in SETS:
        files = list((track3_root / f"{set_name} set").glob("*.mat"))
        assert len(files) > 0, f"No files found for set {set_name}"
        for file in files:
            match = re.match(record_regex, file.name)
            assert match is not None, f"Record {file} does not match expected format"
            subject = match.group("subject")
            bids_path = BIDSPath(
                subject=subject,
                task="imaginedSpeech",
                suffix="eeg",
                datatype="eeg",
                run=SETS[set_name],
            )
            records.append((file, bids_path))

    y_test_path = track3_root / "Test set" / "Track3_Answer Sheet_Test.xlsx"
    df = pd.read_excel(y_test_path)
    df = df.drop(columns=["Unnamed: 0"], index=[0, 1])
    y_test = {f"Data_Sample{i+1:02d}": df.iloc[:, 2 * i + 1].values for i in range(15)}
    return records, y_test


def _read_mat(file_path: Path, suffix, y_test):
    if suffix in ("train", "validation"):
        mat = scipy.io.loadmat(file_path)
        epo = mat[f"epo_{suffix}"]
        mnt = mat["mnt"]
        x = epo["x"][0][0]  # shape (n_times, n_channels, n_epochs)
        x = x.transpose(2, 1, 0)
        y = epo["y"][0][0].argmax(axis=0)
        sfreq = float(epo["fs"][0][0].item())
        ch_names = [str(c.item()) for c in mnt["clab"][0][0][0]]
        assert np.array_equal(
            epo["clab"][0, 0], mnt["clab"][0, 0]
        ), "Channel labels do not match"
    elif suffix == "test":
        mat = h5py.File(file_path, "r")
        epo = mat[f"epo_{suffix}"]
        mnt = mat["mnt"]
        x = np.asarray(epo["x"])  # shape (n_epochs, n_channels, n_times)
        y = y_test[file_path.stem] - 1
        sfreq = float(epo["fs"][0, 0])
        ch_names = [
            "".join(mat[ref[0]][:].astype("uint64").view("U2").flatten())
            for ref in mnt["clab"]
        ]
        ch_names2 = [
            "".join(mat[ref[0]][:].astype("uint64").view("U2").flatten())
            for ref in epo["clab"]
        ]
        assert all(
            c1 == c2 for c1, c2 in zip(ch_names, ch_names2)
        ), "Channel labels do not match"
    else:
        raise ValueError(f"Unknown suffix {suffix}")

    return x, y, sfreq, ch_names


def main(
    source_root: Path,
    bids_root: Path,
    overwrite: bool = False,
    finalize_only: bool = False,
):
    """Convert the CHB-MIT dataset to BIDS format.

    Parameters
    ----------
    source_root : Path
        Path to the root folder
    bids_root : Path
        Path to the root of the BIDS dataset to create.
    overwrite : bool
        If True, overwrite existing BIDS files.
    """
    source_root = Path(source_root).expanduser()
    bids_root = Path(bids_root).expanduser()

    records, y_test = _get_records(source_root)

    # Add bids root:
    bids_root.mkdir(parents=True, exist_ok=True)
    for _, bids_path in records:
        bids_path = bids_path.update(root=bids_root)

    # sanity check: no duplicate bids paths
    bids_paths = [bids_path.fpath for _, bids_path in records]
    assert len(bids_paths) == len(set(bids_paths)), "Duplicate BIDS paths found"

    if finalize_only:
        _finalize_dataset(bids_root, overwrite=overwrite)
        return

    std_list = []
    for source_path, bids_path in records:
        x, y, sfreq, ch_names = _read_mat(
            source_path, EPO_SUFFIX[bids_path.run], y_test
        )

        # ch_pos = mnt['pos_3d'][0,0].T

        description = [LABELS[yi] for yi in y]
        duration = x.shape[-1] / sfreq
        onset = [i * duration for i in range(x.shape[0])]
        annotations = mne.Annotations(
            onset=onset,
            duration=duration,
            description=description,
        )

        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types="eeg",
        )
        info["subject_info"] = {"his_id": bids_path.subject}
        info["description"] = (
            EPO_SUFFIX[bids_path.run] + " set of BCI competition 2020 track 3"
        )

        x_raw = x.transpose(1, 0, 2).reshape(len(ch_names), -1) * 1e-6  # to Volts
        std_list.append(x_raw.std())

        raw = mne.io.RawArray(x_raw, info)
        raw.set_annotations(annotations)

        write_raw_bids(
            raw,
            bids_path,
            overwrite=overwrite,
            verbose=False,
            allow_preload=True,
            format="EDF",
        )

    print(
        f"Overall std: {np.mean(std_list)*1e6:.2f} uV +/- {np.std(std_list)*1e6:.2f} uV"
    )

    _finalize_dataset(bids_root, overwrite=overwrite)


def _finalize_dataset(bids_root: Path, overwrite: bool = False):
    # save script
    script_path = Path(__file__)
    script_dest = bids_root / "code" / script_path.name
    script_dest.parent.mkdir(exist_ok=True)
    shutil.copy2(script_path, script_dest)
    description_file = bids_root / "dataset_description.json"
    if description_file.exists() and overwrite:
        description_file.unlink()
    make_dataset_description(
        path=bids_root,
        name=DATASET_NAME,
        dataset_type="derivative",
        source_datasets=[
            {"URL": "https://osf.io/pq7vb/overview"},
        ],
        authors=["Pierre Guetschel"],
        overwrite=overwrite,
    )

    # cleanup macos hidden files
    for macos_file in bids_root.rglob("._*"):
        macos_file.unlink()

    report_str = make_report(bids_root)
    print(report_str)

    # overwrite README (include automatic report)
    readme_path = bids_root / "README.md"
    with open(readme_path, "w") as f:
        f.write(
            f"# {DATASET_NAME}\n\n{README_CONTENT}\n\n---\n\n"
            f"**Automatic report:**\n\n*Report automatically generated by `mne_bids.make_report()`.*\n\n> {report_str}"
        )

    # Remove participants.json if it exists
    participants_json = bids_root / "participants.json"
    if participants_json.exists():
        participants_json.unlink()
        print(f"Removed {participants_json}")

    # Clean up participants.tsv by removing columns where all values are "n/a"
    participants_tsv = bids_root / "participants.tsv"
    if participants_tsv.exists():
        df = pd.read_csv(participants_tsv, sep="\t")
        # Find columns where all non-participant_id values are "n/a"
        cols_to_drop = []
        for col in df.columns:
            if col != "participant_id" and (df[col] == "n/a").all():
                cols_to_drop.append(col)
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            df.to_csv(participants_tsv, sep="\t", index=False)
            print(
                f"Removed columns with all 'n/a' values from {participants_tsv}: {cols_to_drop}"
            )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
    # python bids_maker/datasets/bcic2020-3.py --source_root ~/data/bcic2020-3/ --bids_root ~/data/bids/bcic2020-3/ --overwrite=True
