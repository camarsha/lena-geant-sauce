import argparse
from tqdm import tqdm
import uproot
import polars as pl
import numpy as np
import awkward as ak
import sys
from pathlib import Path


def add_event(event_list, **kwargs):
    event_list.append(kwargs)


def main():
    parser = argparse.ArgumentParser(
        prog="geant-sauce",
        description="Convert a LENAGe simulation to sauce compatible parquet file.",
        epilog="Author: Caleb Marshall 2025",
    )

    parser.add_argument(
        "root_file", help="ROOT file that has the geant simulation data."
    )
    parser.add_argument(
        "parquet_file",
        help="Name of the output Parquet file. Default will be constructed from ROOT file name.",
        nargs="?",
        default=None,
    )

    parser.add_argument(
        "-c",
        "--hpge_channel",
        help="Channel of the hpge detector.",
        type=int,
        default=0,
    )

    # print help if not enough arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    filename = args.root_file
    if args.parquet_file is None:
        outfile = Path(filename).with_suffix(".parquet")
        outfile_sum = Path(filename + "_sum").with_suffix(".parquet")
    else:
        outfile = args.parquet_file
        outfile_sum = Path(Path(args.parquet_file).stem + "_sum").with_suffix(
            ".parquet"
        )

    r = uproot.open(filename)["fTree;1/RawMC"]
    n_events = len(r["fEventID"].array())
    hpge = r["fEnergyGe"].array().to_numpy() * 1000.0
    hpge_time = ak.fill_none(ak.firsts(r["fGeTime"].array()), 0.0).to_numpy()
    hpge_angle = ak.fill_none(
        ak.firsts(r["fGeCreationDirectionz"].array()), 0.0
    ).to_numpy()
    nai_segs = [
        r[f"fEnergyNaI_seg{i:02}"].array().to_numpy() * 1000.0
        for i in range(16)
    ]
    nai_time = ak.fill_none(ak.firsts(r["fNaITime"].array()), 0.0).to_numpy()

    # this handles the per transition energy deposition.
    # first create a map of the transitions that are present
    sum_data = True
    try:
        transitions = r["fPrimaryGammaE"].array()
        transitions_unique = np.unique(ak.flatten(transitions))
        transition_map = {float(k): v for v, k in enumerate(transitions_unique)}
        hpge_sums = r["fPrimaryGammaEdepGe"].array()
    except uproot.KeyInFileError:
        print("No sum data found.")
        sum_data = False

    events = []
    sum_events = []
    for i in tqdm(range(n_events)):
        if hpge[i] > 0.0:
            add_event(
                events,
                module=225,
                channel=args.hpge_channel,
                adc=hpge[i],
                tdc=hpge_time[i],
                angle=hpge_angle[i],
                evt_ts=i,
            )
            # handle individual transitions.
            if sum_data:
                for t, e in zip(transitions[i], hpge_sums[i]):
                    # we are mapping these to different modules
                    idx = transition_map[float(t)]
                    # this is also a hack for now where the tdc is the transition energy
                    add_event(
                        sum_events,
                        module=225,
                        channel=args.hpge_channel,
                        adc=e,
                        tdc=hpge_time[i],
                        transition_energy=t,
                        transition_idx=idx,
                        angle=hpge_angle[
                            i
                        ],  # needs to be updated at some point.
                        evt_ts=i,
                    )

            for seg_id, seg in enumerate(nai_segs):
                if seg[i] > 0.0:
                    add_event(
                        events,
                        module=112,
                        channel=seg_id,
                        adc=seg[i],
                        tdc=nai_time[i],
                        evt_ts=i,
                    )
    df = pl.DataFrame(events)
    df.write_parquet(outfile)

    df_sum = pl.DataFrame(sum_events)
    df_sum.write_parquet(outfile_sum)


if __name__ == "__main__":
    main()
