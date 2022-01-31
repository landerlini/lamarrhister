from argparse import ArgumentParser
from glob import glob
import pickle
import numpy as np

def merge_hists():
    parser = ArgumentParser(description="Script for merging the histograms produced with lamarrhister")
    parser.add_argument("--histdb", '-H', type=str,
                        help="JSON file listing histograms")
    parser.add_argument("--vardb", '-V', type=str,
                        help="CSV file listing variables")
    parser.add_argument("files", nargs='+', type=str,
                        help="Pickle files with histograms")
    parser.add_argument("--output-filename", '-o', type=str, default="merged.pkl",
                        help="Output file name (Pickle format)")

    args = parser.parse_args()

    merged_dict = None

    files = sum([glob(f) for f in args.files], [])
    print (files)

    for file in files:
        with open(file, 'rb') as f_input:
            hist_data = pickle.load(f_input)

        if merged_dict is None:
            merged_dict = hist_data
        else:
            ## Histograms ##
            hists = []
            for merged_hist, new_hist in zip(merged_dict['hists'], hist_data['hists']):
                # 1D histograms
                if len(merged_hist) == 2:
                    h1, bx1 = merged_hist
                    h2, bx2 = new_hist
                    if np.any(bx1 != bx2):
                        raise ValueError("Inconsistent histograms")
                    hists.append((np.array(h1) + np.array(h2), bx1))

                # 2D histograms
                elif len(merged_hist) == 3:
                    h1, bx1, by1 = merged_hist
                    h2, bx2, by2 = new_hist

                    if np.any(bx1 != bx2) or np.any(by1 != by2):
                        raise ValueError("Inconsistent histograms")
                    hists.append((np.array(h1) + np.array(h2), bx1, by1))

            merged_dict['hists'] = hists

            ## Efficiency Plots ##
            hists = []
            for key in merged_dict['effplots'].keys():
                for cut in merged_dict['effplots'][key].keys():
                    h1, bx1 = merged_dict['effplots'][key][cut]
                    h2, bx2 = hist_data['effplots'][key][cut]
                    if any(bx1 != bx2):
                        raise ValueError(f"Inconsistent histograms {key}")
                    merged_dict['effplots'][key][cut] = (np.array(h1) + np.array(h2), bx1)

    return 0

if __name__ == '__main__':
    exit(merge_hists())


