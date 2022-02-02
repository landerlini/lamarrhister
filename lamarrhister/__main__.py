from argparse import ArgumentParser
import pickle
import json

import numpy as np
import pandas as pd
import uproot


def parse_cut(cut):
    if '|' in cut:
        cuts = cut.replace('||', '|').split('|')
        cut = ' | '.join([f'({c})' for c in cuts])

    if '&' in cut:
        cuts = cut.replace('&&', '&').split('&')
        cut = ' & '.join([f'({c})' for c in cuts])

    if cut[0] != '(' or cut[-1] != ')':
        cut = f"({cut})"

    return cut


def main():
    parser = ArgumentParser(description="Simple python package to fill histograms")
    parser.add_argument("--histdb", '-H', type=str,
                        help="JSON file listing histograms")
    parser.add_argument("--vardb", '-V', type=str,
                        help="CSV file listing variables")
    parser.add_argument("files", nargs='+', type=str,
                        help="ROOT files with nTuples")
    parser.add_argument("--tree", '-t', type=str, default=None,
                        help="Tree name(s) in nTuples")
    parser.add_argument("--naming-scheme", '-n', type=str, default="Lamarr",
                        help="Naming convention for variables, as defined in vardb")
    parser.add_argument("--output-filename", '-o', type=str, default="output.pkl",
                        help="Output file name (Pickle format)")

    args = parser.parse_args()

    vardb = pd.read_csv(args.vardb, engine='python', delimiter=' *, *')
    vardb.set_index('Variable', inplace=True)
    var_code = {k: str(v.values[0]) for k, v in vardb[[args.naming_scheme]].iterrows()}
    var_code = {k: (None if v == 'None' else v) for k,v in var_code.items()}


    with open(args.histdb, 'r') as f:
        histdb = json.load(f)

    selection = histdb['selection'][args.naming_scheme]

    output_dict = dict(hists=[], effplots=dict())

    for file_name in args.files:
        root_file = uproot.open(file_name)
        trees = [t for t in root_file.keys() if "TTree" in root_file[t].__class__.__name__]

        tree_name = args.tree
        if len(trees) == 0:
            raise ValueError(f"Found no tree in file {file_name}")
        elif len(trees) == 1:
            tree_name = next(iter(trees))

        if tree_name is None:
            raise ValueError(f"Multiple trees in {file_name}: {trees}.\n" +
                             "Specify one with --tree")

        parsed_selection = [parse_cut(cut) for cut in selection]
        selection_string = " & ".join(parsed_selection)

        print (selection_string)

        df = uproot.open(file_name)[tree_name].arrays(library='np', cut=selection_string)
        df = pd.DataFrame({k: v for k, v in df.items() if len(v.shape) == 1})

        for histogram in histdb['hists']:
            if 'selection' in histogram and histogram['selection'] != '':
                hist_df = df.query(histogram['selection'].format(**var_code))
            else:
                hist_df = df

            weight = hist_df.eval(var_code['weight']) if histogram['weight'] else None
            variables = histogram['vars']
            if len(variables) == 1:  # 1D histogram
                var = variables[0]
                low, high, nBins = vardb.loc[var, ['Min', 'Max', 'nBins']]
                binning = np.linspace(low, high, nBins+1)
                output_dict['hists'].append(
                    np.histogram(hist_df.eval(var_code[var]),
                                 bins=binning,
                                 weights=weight)
                )
            elif len(variables) == 2: # 2D histogram
                binning = []
                for var in variables:
                    low, high, nBins = vardb.loc[var, ['Min', 'Max', 'nBins']]
                    binning.append(np.linspace(low, high, nBins+1))

                x, y = hist_df.eval([var_code[v] for v in variables])
                output_dict['hists'].append(
                    np.histogram2d(
                        x.values, y.values,
                        bins=binning,
                        weights=weight
                    )
                )

        for effplot in histdb['effplots']:
            data = dict()
            output_dict['effplots'][effplot['name']] = data

            if 'selection' in effplot.keys():
                hist_df = df.query(effplot['selection'].format(**var_code))
            else:
                hist_df = df

            weight = hist_df.eval(var_code['weight']) if effplot['weight'] else None
            var = hist_df.eval(var_code[effplot['var']]).values
            low, high, nBins = vardb.loc[effplot['var'], ['Min', 'Max', 'nBins']]
            binning = np.linspace(low, high, nBins + 1)
            data['full'] = np.histogram(var, bins=binning, weights=weight)
            for cut in effplot['cuts']:
                sel_df = hist_df.query(cut.format(**var_code))
                var = sel_df.eval(var_code[effplot['var']]).values
                weight = sel_df.eval(var_code['weight']) if effplot['weight'] else None

                data[cut] = np.histogram(var, bins=binning, weights=weight)

    with open(args.output_filename, 'wb') as f_output:
        pickle.dump(output_dict, f_output)
    print(f"Output file {args.output_filename} was stored on disk")

    return 0

if __name__ == '__main__':
    exit(main())
