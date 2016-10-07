import numpy as np
import pandas as pd
import csv
import re

def parse_xgb_dump(dumpfile):
    boos = -1
    feat = '---'
    feat_re = re.compile('(\\d*):.([0-9A-Za-z_]+)<([0-9-.]+)\].yes=([0-9]+).no=([0-9]+).missing=([0-9]+).gain=([0-9.]+).cover=([0-9.]+)')
    leaf_re = re.compile('(\\d*).leaf=([0-9.e-]+).cover=([0-9.]+)')
    headers = ['id', 'Feature', 'Split', 'Yes', 'No', 'Missing', 'Gain', 'Cover', 'Tree']
    res = []
    with open (dumpfile, 'rb') as f:
        reader = csv.reader(f, delimiter='*', quotechar='"')
        for line in reader:
            l = line[0].strip()
            if l.startswith('booster'):
                boos = re.findall('[0-9]+', l)[0]
            else:
                if 'leaf' in l:
                    iid, lgain, lcov = re.findall(leaf_re, l)[0]
                    feat = (iid, 'leaf', None, None, None, None, lgain, lcov)
                else:
                    feat = re.findall(feat_re, l)[0]
                iid, feat, spl, y, n, miss, fgain, fcov = feat
                res.append([boos + '-' + iid, feat, spl, y, n, miss, fgain, fcov, boos])
    res = pd.DataFrame(res, columns=headers, dtype=np.float)
    res_gr = res[res.Feature != 'leaf'].groupby('Feature')
    res_gr = res_gr['Gain', 'Cover'].sum()
    res_gr['GainShare'] = res_gr['Gain'] / res_gr['Gain'].sum()
    res_gr['CoverShare'] = res_gr['Cover'] / res_gr['Cover'].sum()
    res_gr = res_gr.sort_values(by='GainShare', ascending=False).reset_index()
    return res_gr, res