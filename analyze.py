"""Input diagnostics. Keep valid long tails; IQR filtering is a sensitivity case only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from prepare import ROOT, read_csv, save_csv, save_json

os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FAMILIES = {'gamma': (stats.gamma, 2, {'floc': 0}),
            'lognorm': (stats.lognorm, 2, {'floc': 0}),
            'expon': (stats.expon, 1, {'floc': 0}),
            'norm': (stats.norm, 2, {}),
            'uniform': (stats.uniform, 2, {})}


def verify_data(data_dir):
    for name, digest in json.loads((Path(data_dir) / 'manifest.json').read_text()).items():
        if hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Prepared data changed: {name}. Rebuild or document a new source version.')


def samples(data_dir):
    verify_data(data_dir)
    grouped = defaultdict(list)
    for row in read_csv(Path(data_dir) / 'observations.csv'):
        grouped[row['dataset']].append(row)
    return grouped


def valid_values(rows):
    return np.asarray([float(r['value']) for r in rows if r['status'] == 'valid'], dtype=float)


def iqr_mask(x, multiplier=1.5):
    x = np.asarray(x, dtype=float)
    if not len(x):
        raise ValueError('Empty sample')
    q1, q3 = np.quantile(x, [0.25, 0.75], method='linear')
    low, high = q1 - multiplier*(q3-q1), q3 + multiplier*(q3-q1)
    return (x >= low) & (x <= high), float(low), float(high)


def plotting_positions(n):
    if n < 1: raise ValueError('Empty sample')
    return (np.arange(1, n+1)-0.5)/n


def describe(x):
    return {'n':len(x), 'zero_count':int(np.sum(x == 0)), 'minimum':float(np.min(x)),
            'mean':float(np.mean(x)), 'median':float(np.median(x)),
            'sd':float(np.std(x, ddof=1)) if len(x)>1 else None,
            'p90':float(np.quantile(x,0.9)), 'p95':float(np.quantile(x,0.95)), 'maximum':float(np.max(x))}


def acf_observations(x, lags=10):
    x = np.asarray(x)-np.mean(x)
    denominator = np.dot(x, x)
    if denominator == 0: return []
    return [float(np.dot(x[:-k], x[k:])/denominator) for k in range(1,min(lags,len(x)-1)+1)]


def fit_sample(x, family):
    dist, k, kwargs = FAMILIES[family]
    params = tuple(float(p) for p in dist.fit(x, **kwargs))
    log_likelihood = float(np.sum(dist.logpdf(x, *params)))
    if not np.isfinite(log_likelihood): raise ValueError('Non-finite fitted likelihood')
    return dist, params, {'parameters':params, 'parameter_order':
        ['shape','loc_seconds','scale_seconds'] if family in ['gamma','lognorm'] else ['loc_seconds','scale_seconds'],
        'n_parameters':k, 'log_likelihood':log_likelihood, 'aic':2*k-2*log_likelihood,
        'bic':k*np.log(len(x))-2*log_likelihood,
        'negative_duration_probability':float(dist.cdf(0, *params)),
        'ks_distance':float(stats.kstest(x, dist.cdf, args=params).statistic),
        'note':'Uniform has parameter-dependent support; its AIC is descriptive, not automatic model selection.' if family=='uniform' else ''}


def refit_bootstrap_ks(x, family, repetitions, seed):
    """Parametric bootstrap refits each sample; results are conditional on IID continuous data."""
    dist, params, fitted = fit_sample(x, family)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(repetitions):
        generated = dist.rvs(*params, size=len(x), random_state=rng)
        _, _, result = fit_sample(generated, family)
        exceed += result['ks_distance'] >= fitted['ks_distance']
    return (exceed+1)/(repetitions+1)


def analyze(data_dir, output, bootstrap=199, seed=20261016):
    if bootstrap < 0: raise ValueError('bootstrap must be nonnegative')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    grouped=samples(data_dir); summary={}; fits={}; flagged=[]
    for dataset, rows in sorted(grouped.items()):
        x=valid_values(rows)
        if not len(x): continue
        is_duration=rows[0]['unit']=='seconds'
        entry=describe(x)
        entry.update({'total_records':len(rows),'invalid_records':len(rows)-len(x),'unit':rows[0]['unit'],
                      'acf_by_retained_source_order':acf_observations(x),
                      'acf_note':'Exploratory source-order correlation; incomplete records/unknown observation order prevent an independence claim.'})
        if not is_duration:
            # Wilson interval is descriptive under IID Bernoulli assumption, not adjusted for sampling design.
            interval=stats.binomtest(int(x.sum()),len(x)).proportion_ci(method='wilson')
            entry['wilson95_iid_assumption']=[float(interval.low),float(interval.high)]
            summary[dataset]=entry;continue
        mask,low,high=iqr_mask(x)
        entry.update({'iqr_low':low,'iqr_high':high,'iqr_flagged':int((~mask).sum()),
                      'iqr_sensitivity':describe(x[mask])})
        for row,keep in zip([r for r in rows if r['status']=='valid'],mask):
            if not keep: flagged.append({'dataset':dataset,'sample_id':row['sample_id'],'value_seconds':row['value'],
                'source_file':row['source_file'],'source_cells':row['source_cells'],'rule':'outside Q1-1.5IQR or Q3+1.5IQR; linear quantiles',
                'baseline_action':'retained','sensitivity_action':'excluded; not a confirmed error'})
        summary[dataset]=entry
        # Do not quietly remove rounded zero intervals to fit a continuous positive distribution.
        if len(x)<5 or np.any(x<=0) or np.ptp(x)==0:
            fits[dataset]={'status':'empirical only: small/constant sample or zero-valued observations'}
            continue
        fits[dataset]={}
        for case,values in [('all_valid',x),('iqr_sensitivity',x[mask])]:
            fits[dataset][case]={}
            for family in FAMILIES:
                dist,params,result=fit_sample(values,family)
                result['qq_finite']=bool(np.isfinite(dist.ppf(plotting_positions(len(values)),*params)).all())
                # Avoid post-selection testing after filtering. Original observed sequences may be dependent.
                if bootstrap and case=='all_valid' and dataset in ['order_completion','dining_duration','kiosk_notebook'] and family!='uniform':
                    family_seed=seed+int.from_bytes(hashlib.sha256((dataset+family).encode()).digest()[:4],'big')
                    result['bootstrap_ks_p_iid_continuous_assumption']=refit_bootstrap_ks(values,family,bootstrap,family_seed)
                    result['bootstrap_repetitions']=bootstrap
                    result['bootstrap_seed']=family_seed
                    result['bootstrap_scope']='Exploratory only: rounding, dependence and selection of complete records are not modeled.'
                fits[dataset][case][family]=result
        if dataset in ['order_completion','dining_duration','kiosk_notebook','kiosk_service_saturday','seat_delay_notebook','call2_elapsed']:
            fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
            axes[0,0].hist(x,bins='auto',color='#397693',alpha=.8)
            axes[0,0].axvline(high,color='#b1442d',linestyle='--',label='Q3 + 1.5 IQR (flag only)')
            axes[0,0].set(xlabel='Seconds',ylabel='Count',title='All valid observations');axes[0,0].legend(fontsize=8)
            axes[0,1].step(np.sort(x),np.arange(1,len(x)+1)/len(x),where='post',label='All valid')
            axes[0,1].step(np.sort(x[mask]),np.arange(1,mask.sum()+1)/mask.sum(),where='post',label='IQR sensitivity')
            axes[0,1].set(xlabel='Seconds',ylabel='ECDF');axes[0,1].legend()
            dist,params,_=fit_sample(x,'gamma');q=dist.ppf(plotting_positions(len(x)),*params)
            axes[1,0].scatter(q,np.sort(x),s=14);limits=[min(q.min(),x.min()),max(q.max(),x.max())]
            axes[1,0].plot(limits,limits,color='#b1442d');axes[1,0].set(xlabel='Gamma theoretical seconds',ylabel='Observed seconds',title='Q-Q: finite plotting positions')
            a=acf_observations(x);axes[1,1].bar(np.arange(1,len(a)+1),a,color='#397693')
            axes[1,1].set(xlabel='Lag in retained source order',ylabel='Sample ACF',title='Observed values, not synthetic draws')
            fig.suptitle(f'{dataset} | n={len(x)} | IQR flags={int((~mask).sum())}')
            fig.savefig(output/f'{dataset}.png',dpi=150);plt.close(fig)
    save_json(output/'summary.json',summary);save_json(output/'fits.json',fits)
    save_csv(output/'iqr_flags.csv',flagged,['dataset','sample_id','value_seconds','source_file','source_cells','rule','baseline_action','sensitivity_action'])
    save_json(output/'analysis_run.json',{'seed':seed,'bootstrap_repetitions':bootstrap,
        'baseline':'all valid observations; no long-tail deletion',
        'iqr':'1.5 times IQR, numpy linear quantiles, both tails, per dataset',
        'data_manifest':json.loads((Path(data_dir)/'manifest.json').read_text()),
        'source':'https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.goodness_of_fit.html'})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'results/input')
    p.add_argument('--bootstrap',type=int,default=199);p.add_argument('--seed',type=int,default=20261016)
    a=p.parse_args();summary=analyze(a.data,a.output,a.bootstrap,a.seed)
    print(f'Analyzed {len(summary)} datasets; output: {a.output}')
