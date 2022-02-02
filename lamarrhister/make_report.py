from argparse import ArgumentParser
from pprint import pprint
from datetime import datetime
import os.path
import json
import pickle
import re

import numpy as np
from scipy.special import binom
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import pandas as pd
from html_reports import Report


def efficiency(n, k):
    """
    Efficiency confidence interval given n and k

    :param n: int
        * Total number of entries
    :param k: int
        * Number of selected entries
    :return: tuple(float, float, float)
        * (lower confidence bound, mean efficiency, higher confidence bound)
    """
    eff = np.linspace(0, 1, 1000)
    p = binom(n, k) * (eff ** k) * (1 - eff) ** (n - k)
    cumulative_p = np.cumsum(p)
    cumulative_p /= cumulative_p[-1]

    if k == 0:
        low, high = 0., np.interp(0.32, cumulative_p, eff)
    elif k == n:
        low, high = np.interp(1.-0.32, cumulative_p, eff), 1.
    else:
        low, high = np.interp([0.16, 1 - 0.16], cumulative_p, eff)

    return min(low, k / n), k / n, max(high, k / n)


def greeks(string, fmt):
    if fmt.lower() == 'html':
        string = re.sub("#([A-Za-z]*)([^A-Z-a-z])", r"&\1;\2", string)
        string = re.sub("#([A-Za-z]*)$", r"&\1;", string)
        string = re.sub("_\{([a-zA-Z0-9]*)\}", r"<sub>\1</sub>", string)
        string = re.sub("\^\{([+\-a-zA-Z0-9]*)\}", r"<sup>\1</sup>", string)
    elif fmt.lower() == 'latex':
        string = re.sub("#([A-Za-z]*)([^A-Z-a-z])", r"\\\1\2", string)
        string = re.sub("#([A-Za-z]*)$", r"\\\1", string)
        string = " ".join([f"${a}$" if '_' in a or '\\' in a or '?' in a else a for a in string.split(' ')])

    return string

def draw_histogram (boundaries, contentsL, contentsR, title):
    if np.sum(contentsR) == 0 or np.sum(contentsL) == 0:
        return False

    contentsR = contentsR * np.sum(contentsL) / np.sum(contentsR)
    plt.figure(figsize=(5, 3.5), dpi=100)
    x = (boundaries[1:] + boundaries[:-1]) / 2
    dx = (boundaries[1:] - boundaries[:-1]) / 2
    plt.hist(x, bins=boundaries, weights=contentsR, alpha=0.2, color='#08c')
    plt.hist(x, bins=boundaries, weights=contentsR,
             alpha=1, color='#08c', histtype='step', linewidth=2, label='Reference'
             )

    plt.subplots_adjust(top=0.8, bottom=0.15, right=0.95, left=0.15)
    msk = (contentsL > 0)
    plt.errorbar(x[msk], contentsL[msk], np.sqrt(contentsL[msk]), dx[msk],
                 fmt='o', color='#c08', markersize=3,
                 label='GAN Simulation')
    plt.legend(title=title, ncol=2, bbox_to_anchor=(0.67, 1.27),
               title_fontsize='large', loc='upper center', framealpha=1, shadow=True)

    plt.text(0.02, 1.05, "LHCb\nSimulation", transform=plt.gca().transAxes, fontfamily='serif', fontsize='large')
    plt.ylabel('Normalized candidates')
    return True


def make_report():
    parser = ArgumentParser(description="Produce an HTML Report comparing two series of histograms")
    parser.add_argument("--title", '-t', type=str, default="Validation", help="Report title")
    parser.add_argument("--histdb", '-H', type=str,
                        help="JSON file listing histograms")
    parser.add_argument("--vardb", '-V', type=str,
                        help="CSV file listing variables")
    parser.add_argument("--lamarr", '-L', type=str,
                        help="Pickle file with merged Lamarr histograms")
    parser.add_argument("--reference", '-R', type=str,
                        help="Pickle file with merged reference (TurCal or Detailed Sim) histograms")
    parser.add_argument("--output-filename", '-o', type=str, default="report.html",
                        help="Output report file name (HTML format)")

    args = parser.parse_args()

    with open(args.histdb, 'rb') as file_in:
        histdb = json.load(file_in)

    vardb = pd.read_csv(args.vardb, engine='python', delimiter=' *, *')
    var_title = {k: v for k, v in vardb[['Variable', 'Title']].values}

    with open(args.lamarr, 'rb') as file_in:
        lamarr = pickle.load(file_in)
    with open(args.reference, 'rb') as file_in:
        reference = pickle.load(file_in)

    report = Report()
    report.add_markdown(f"## Report {args.title}")
    report.add_markdown(
        f"Generated at **{datetime.now()}** and stored at `{os.path.abspath(args.output_filename)}`"
    )
    report.add_markdown('### Applied Selection')
    for key, value in histdb['selection'].items():
        report.add_markdown(f'#### {key.capitalize()}\n * ' + "\n * ".join(histdb['selection'][key]))

    report.add_markdown("### Histograms")
    for hist_desc, histR, histL in zip(histdb['hists'], reference['hists'], lamarr['hists']):
        report.add_markdown(f'#### {" #perp ".join(var_title[v] for v in hist_desc["vars"])}')
        if len(hist_desc['vars']) == 1:  ## 1D histogram
          if False:
            contentsL, boundaries = histL
            contentsR, _ = histR
            draw_histogram(boundaries, contentsL, contentsR, title=greeks(histdb['title'], 'latex'))
            plt.xlabel(greeks(var_title[hist_desc['vars'][0]], 'latex'))
            report.add_figure()
            plt.close()
        elif len(hist_desc['vars']) == 2:  ## 2D histogram
            contentsL, boundariesX, boundariesY = histL
            contentsR, _, _ = histR
            if len(boundariesY) < 20:
              if False:
                for iRow, (low, high) in enumerate(zip(boundariesY[:-1], boundariesY[1:])):
                    if draw_histogram(boundariesX, contentsL[:,iRow], contentsR[:,iRow],
                                   title=greeks(histdb['title'], 'latex')):
                        plt.xlabel(greeks(var_title[hist_desc['vars'][0]], 'latex'))
                        bin_string = greeks(var_title[hist_desc['vars'][1]], 'latex')
                        if 'MeV' in bin_string:
                            bin_string = f"{bin_string} in [{low:.0f}, {high:.0f}]"
                        elif 'GeV' in bin_string:
                            bin_string = f"{bin_string} in [{low:.1f}, {high:.1f}]"
                        elif '\eta' in bin_string:
                            bin_string = f"{bin_string} in [{low:.1f}, {high:.1f}]"

                        if len(bin_string) > 25:
                            tkns = bin_string.split(' ')
                            bin_string = " ".join(tkns[:len(tkns)//2] + ['\n'] + tkns[len(tkns)//2:])
                        plt.text(0.98, 0.95, bin_string,
                                 ha='right', va='top',
                                 transform=plt.gca().transAxes,
                                 fontfamily='serif',
                                 fontsize='medium')
                        report.add_figure()
                        plt.close()
            else: # 2D scatter plots
                plt.figure(figsize=(5, 3.5), dpi=100)
                x = (boundariesX[1:] + boundariesX[:-1])/2
                y = (boundariesY[1:] + boundariesY[:-1])/2
                x, y = np.meshgrid(x, y)
                report.add_markdown(f"Shape: {x.shape}")
                contentsR *= 50 / np.max(contentsR)
                contentsL *= 50 / np.max(contentsL)
                plt.scatter(x.flatten(), y.flatten(), s=contentsR.T.flatten(), alpha=0.5, c='#08c', marker='s', label='Reference')
                plt.scatter(x.flatten(), y.flatten(), s=contentsL.T.flatten(), alpha=0.5, c='#c08', marker='s', label='GAN Simulation')

                plt.xlabel(greeks(var_title[hist_desc['vars'][0]], 'latex'))
                plt.ylabel(greeks(var_title[hist_desc['vars'][1]], 'latex'))

                plt.subplots_adjust(top=0.8, bottom=0.15, right=0.95, left=0.15)

                title = greeks(histdb['title'], 'latex')
                plt.legend(title=title, ncol=2, bbox_to_anchor=(0.67, 1.27),
                           title_fontsize='large', loc='upper center', framealpha=1, shadow=True)

                plt.text(0.02, 1.05, "LHCb\nSimulation", transform=plt.gca().transAxes, fontfamily='serif',
                         fontsize='large')

                report.add_figure()
                plt.close()

    report.add_markdown("### Efficiency plots")
    pprint(histdb['effplots'])


    for effplot in histdb['effplots']:
      if False:
        ref = reference['effplots'][effplot['name']]
        lam = lamarr['effplots'][effplot['name']]

        report.add_markdown(
            f' * {effplot["name"]}' +
            f"   [{', '.join([r for r in lam.keys() if r != 'full'])}]"
            )

        ref_deno, boundaries = ref['full']
        lam_deno, _ = lam['full']

        for cut in [k for k in lam.keys() if k != 'full']:
            ref_nume, _ = ref[cut]
            lam_nume, _ = ref[cut]
            x = (boundaries[1:] + boundaries[:-1])/2
            mskr = (ref_deno > 0)
            mskl = (lam_deno > 0)
            xerr = (boundaries[1:] - boundaries[:-1])/2
            ref_eff = np.array(
                [efficiency(n, k) for n, k in np.c_[ref_deno[mskr], ref_nume[mskl]]]
            )
            lam_eff = np.array(
                [efficiency(n, k) for n, k in np.c_[lam_deno[mskr], lam_nume[mskl]]]
            )

            plt.figure(figsize=(5, 3.5), dpi=100)
            ref_y = 0.5*(ref_eff[:, 0] + ref_eff[:, 2])
            ref_yerr = [ref_y - ref_eff[:, 0], ref_eff[:, 2] - ref_y]
            plt.errorbar(x[mskr], ref_y, ref_yerr, xerr[mskr],
                         fmt='o', color='#6be', markersize=0, linewidth=7, alpha=1,
                         label='Reference'
                         )
            lam_y = lam_eff[:, 1]
            lam_yerr = [lam_y - lam_eff[:, 0], lam_eff[:, 2] - lam_y]
            plt.errorbar(x[mskl], lam_y, lam_yerr, xerr[mskl],
                         fmt='o', color='#c08', markersize=4,
                         label='GAN Simulation')

        plt.ylim(-0.1, 1.1)

        plt.subplots_adjust(top=0.8, bottom=0.15, right=0.95, left=0.15)

        title = greeks(histdb['title'], 'latex')
        plt.legend(title=title, ncol=2, bbox_to_anchor=(0.67, 1.27),
                   title_fontsize='large', loc='upper center', framealpha=1, shadow=True)

        plt.text(0.02, 1.05, "LHCb\nSimulation", transform=plt.gca().transAxes, fontfamily='serif',
                 fontsize='large')

        plt.xlabel(var_title[effplot['var']])
        plt.ylabel("Selection efficiency")

        report.add_figure()
        plt.close()

        #report.add_markdown(f"    {', '.join([r.format(**var_title) for r in ref.keys() if r != 'full'])}")




    report.write_report(filename=args.output_filename)

    with open(args.output_filename, 'r') as file_in:
        raw_html = file_in.read()

    with open(args.output_filename, 'w', encoding='utf-8') as file_out:
        file_out.write(greeks(raw_html, 'HTML'))





    return 0

if __name__ == '__main__':
    exit(make_report())
