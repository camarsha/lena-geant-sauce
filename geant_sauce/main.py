import argparse
from numpy import append
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
        default=None,
    )

    # print help if not enough arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    filename = args.root_file
    if args.parquet_file is None:
        outfile = Path(filename).with_suffix(".parquet")
    else:
        outfile = args.parquet_file
    
    r = uproot.open(filename)["fTree;1/RawMC"]
    n_events = len(r["fEventID"].array())
    hpge = r["fEnergyGe"].array().to_numpy() * 1000.0
    hpge_time = ak.fill_none(ak.firsts(r["fGeTime"].array()), 0.0).to_numpy()
    nai_segs = [
        r[f"fEnergyNaI_seg{i:02}"].array().to_numpy() * 1000.0 for i in range(16)
    ]
    nai_time = ak.fill_none(ak.firsts(r["fNaITime"].array()), 0.0).to_numpy()
    events = []
    for i in tqdm(range(n_events)):
        if hpge[i] > 0.0:
            add_event(
                events,
                module=225,
                channel=0,
                adc=hpge[i],
                tdc=hpge_time[i],
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
    df.write_parquet("test.parquet")


if __name__ == "__main__":
    main()
