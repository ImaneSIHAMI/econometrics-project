"""Additional geographical and robustness analysis for the diesel-price study.

Run `python run_analysis.py` AFTER `python run_all.py`.
No original download/parse scripts or raw/interim Parquet files are modified.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from . import pipeline
from .geography import attach_geography, REGIONS

ROOT=pipeline.ROOT

def csv(df, name, directory):
    pipeline.write_csv(df, directory/name)


def station_geography(cfg, tables):
    """Station-year geographical/characteristic records from original annual metadata."""
    frames=[]
    for y in range(cfg['analysis_start_year'],cfg['analysis_end_year']+1):
        p=ROOT/'data'/'interim'/f'stations_{y}.parquet'
        if not p.exists():
            raise FileNotFoundError(f'Missing {p}; run src/parse_data.py first')
        d=pd.read_parquet(p,columns=['station_id','postal_code','road_type','automate_24_24','n_services'])
        d['year']=y
        dup=d.duplicated('station_id',keep=False)
        n_conflicts=(d.loc[dup].groupby('station_id')[['postal_code','road_type','automate_24_24','n_services']]
                     .nunique(dropna=False).gt(1).any(axis=1).sum()) if dup.any() else 0
        d=d.drop_duplicates('station_id',keep='first').copy()
        d['motorway']=d.road_type.astype('string').str.upper().map({'A':1.0,'R':0.0})
        d['automated']=d.automate_24_24.map({True:1.0,False:0.0})
        d['n_services']=pd.to_numeric(d.n_services,errors='coerce')
        d=attach_geography(d)
        frames.append(d)
        csv(pd.DataFrame([{
            'year':y,'station_metadata_rows':len(d),'conflicting_duplicate_station_keys':int(n_conflicts),
            'unmapped_region':int(d.region_code.isna().sum()),
            'unknown_motorway':int(d.motorway.isna().sum()),
            'missing_service_count':int(d.n_services.isna().sum()),
            'missing_automation':int(d.automated.isna().sum()),
            'motorway_stations':int((d.motorway==1).sum()),
            'automated_stations':int((d.automated==1).sum())
        }]),f'metadata_quality_{y}.csv',tables)
    meta=pd.concat(frames,ignore_index=True)
    csv(meta.groupby(['year','geography_status'],dropna=False).size().rename('station_years').reset_index(),
        'geography_assignment_audit.csv',tables)
    csv(meta.groupby(['year','region_code','region_name'],dropna=False).agg(
        stations=('station_id','nunique'),motorway_stations=('motorway',lambda s:int((s==1).sum())),
        automated_stations=('automated',lambda s:int((s==1).sum()))).reset_index(),
        'station_coverage_by_region_year.csv',tables)
    # Changing characteristics for the same station across annual metadata files.
    variability=(meta.groupby('station_id').agg(years=('year','nunique'),
        motorway_values=('motorway',lambda x:x.nunique(dropna=True)),
        automation_values=('automated',lambda x:x.nunique(dropna=True)),
        service_values=('n_services',lambda x:x.nunique(dropna=True)),
        postal_values=('postal_code',lambda x:x.nunique(dropna=True))).reset_index())
    csv(variability,'station_characteristics_stability.csv',tables)
    return meta


def join_monthly(monthly,meta):
    d=monthly.drop(columns=['region_code','region_name','geography_status','postal_code'],errors='ignore').copy()
    geo=meta[['station_id','year','postal_code','region_code','region_name','geography_status']].copy()
    return d.merge(geo,on=['station_id','year'],how='left',validate='many_to_one')


def monthly_from_weekly(weekly,cfg):
    """Same aggregation rule as original pipeline; no overwrite of original panel."""
    d=(weekly.groupby(['station_id','month'],sort=False,observed=True)
       .agg(price=('price','mean'),weeks_observed=('price','size'),year=('year','first'),
            motorway=('motorway','first'),n_services=('n_services','first'),
            automated=('automated','first'),department_code=('department_code','first'))
       .reset_index())
    return d[d.weeks_observed>=cfg['min_weekly_observations_per_month']].copy()


def descriptive(monthly,tables,figures):
    mapped=monthly.dropna(subset=['region_code','price']).copy()
    cov=monthly.groupby('year',sort=True).agg(station_months=('price','size'),
        stations=('station_id','nunique'),months=('month','nunique'),
        unmapped_station_months=('region_code',lambda x:int(x.isna().sum()))).reset_index()
    csv(cov,'panel_coverage_by_year.csv',tables)
    reg=(mapped.groupby(['region_code','region_name'],sort=True).agg(
        mean_price=('price','mean'),median_price=('price','median'),
        sd_price=('price','std'),station_months=('price','size'),
        stations=('station_id','nunique')).reset_index())
    csv(reg,'region_price_summary.csv',tables)
    by_year=(mapped.groupby(['year','region_code','region_name']).agg(
        mean_price=('price','mean'),station_months=('price','size'),
        stations=('station_id','nunique')).reset_index())
    csv(by_year,'region_price_by_year.csv',tables)
    # Within-region, same-month dispersion is more comparable than pooled levels.
    disp=(mapped.groupby(['region_code','region_name','month']).agg(
        n_stations=('station_id','nunique'),sd_across_stations=('price','std'),
        p10=('price',lambda x:x.quantile(.1)),p90=('price',lambda x:x.quantile(.9)),
        mean_price=('price','mean')).reset_index())
    disp['p90_minus_p10']=disp.p90-disp.p10
    csv(disp,'within_region_monthly_dispersion.csv',tables)
    spread=(disp.groupby(['region_code','region_name']).agg(
        mean_within_month_sd=('sd_across_stations','mean'),
        mean_within_month_p90_p10=('p90_minus_p10','mean'),
        region_months=('month','nunique')).reset_index())
    csv(spread,'within_region_dispersion_summary.csv',tables)
    # Month-adjusted region difference, descriptive, not a causal estimate.
    national=mapped.groupby('month').price.mean().rename('national_observed_mean')
    regmo=(mapped.groupby(['region_code','region_name','month']).price.mean().rename('region_mean').reset_index()
           .join(national,on='month'))
    regmo['region_minus_observed_national']=regmo.region_mean-regmo.national_observed_mean
    adjusted=regmo.groupby(['region_code','region_name']).agg(
        average_monthly_gap=('region_minus_observed_national','mean'),
        months=('month','nunique')).reset_index().sort_values('average_monthly_gap')
    csv(adjusted,'region_month_adjusted_price_gaps.csv',tables)
    if len(adjusted):
        fig,ax=plt.subplots(figsize=(10,7))
        ax.barh(adjusted.region_name,adjusted.average_monthly_gap*100)
        ax.axvline(0,color='black',linewidth=.8)
        ax.set(xlabel='Mean region minus observed national monthly mean (cents/L)',
               title='Geographical differences after descriptive month adjustment')
        fig.tight_layout();fig.savefig(figures/'07_region_month_adjusted_differences.png',dpi=140);plt.close(fig)
    if len(spread):
        s=spread.sort_values('mean_within_month_p90_p10')
        fig,ax=plt.subplots(figsize=(10,7))
        ax.barh(s.region_name,s.mean_within_month_p90_p10*100)
        ax.set(xlabel='Average within-region monthly P90–P10 (cents/L)',
               title='Price dispersion between stations in the same region and month')
        fig.tight_layout();fig.savefig(figures/'08_within_region_station_dispersion.png',dpi=140);plt.close(fig)
    if len(by_year):
        mat=by_year.pivot(index='region_name',columns='year',values='mean_price').sort_index()
        fig,ax=plt.subplots(figsize=(9,7))
        im=ax.imshow(mat.to_numpy(),aspect='auto',cmap='viridis')
        ax.set_yticks(np.arange(len(mat.index)),labels=mat.index,fontsize=8)
        ax.set_xticks(np.arange(len(mat.columns)),labels=mat.columns)
        ax.set(title='Mean diesel price by region and year (EUR/L)')
        fig.colorbar(im,ax=ax,label='EUR/L',shrink=.7)
        fig.tight_layout();fig.savefig(figures/'09_region_year_heatmap.png',dpi=140);plt.close(fig)
    return cov,reg,spread,adjusted


def select_station_sample(d,cap,seed):
    """Deterministic whole-station selection; cap=0 selects all stations."""
    if not cap or len(d)<=cap:
        return d.copy(),False
    sizes=d.station_id.value_counts()
    order=d.station_id.drop_duplicates().sample(frac=1,random_state=seed)
    selected=[];n=0
    for station in order:
        count=int(sizes[station])
        if n+count<=cap:
            selected.append(station);n+=count
    return d[d.station_id.isin(selected)].copy(),True


def regression_complete_cases(monthly):
    d=monthly.dropna(subset=['price','motorway','n_services','automated','region_code','month','station_id']).copy()
    d=d[(d.price>0) & d.region_code.astype(str).isin(REGIONS)]
    return d


def fit_region_model(d, label, tables):
    """Region + calendar-month FE; station-clustered OLS; FWL time absorption."""
    if len(d)<100 or d.station_id.nunique()<10:
        raise ValueError(f'{label}: insufficient station-month observations')
    d=d.reset_index(drop=True)
    X=d[['motorway','n_services','automated']].astype(float).reset_index(drop=True)
    geo=pd.get_dummies(d.region_code.astype(str),prefix='region',drop_first=True,dtype=float)
    X=pd.concat([X,geo],axis=1)
    y=d.price.astype(float).reset_index(drop=True)
    X=X-X.groupby(d.month).transform('mean')
    yr=y-y.groupby(d.month).transform('mean')
    X=X.loc[:,X.std().fillna(0)>1e-12]
    # Keep coefficient estimates and covariance computation efficient on large laptops.
    fit=sm.OLS(yr.to_numpy(),X.to_numpy()).fit(
        cov_type='cluster',cov_kwds={'groups':d.station_id.astype(str).to_numpy()},use_t=False)
    rss=float(np.square(fit.resid).sum()); tss=float(np.square(y-y.mean()).sum())
    results=[]
    for i,term in enumerate(X.columns):
        results.append({'specification':label,'term':term,'coef_eur_per_litre':float(fit.params[i]),
            'se_cluster_station':float(fit.bse[i]),'p_value':float(fit.pvalues[i]),
            'ci_lower_95':float(fit.conf_int()[i,0]),'ci_upper_95':float(fit.conf_int()[i,1]),
            'station_months':int(len(d)),'stations':int(d.station_id.nunique()),
            'overall_r2':1-rss/tss if tss else np.nan,
            'region_controls':'postal-derived administrative region',
            'time_controls':'calendar-month fixed effects'})
    csv(pd.DataFrame(results),f'{label}_all_coefficients.csv',tables)
    return pd.DataFrame(results)


def sample_comparison(full,selected,label,tables):
    def describe(d,sample):
        return {'sample':sample,'station_months':len(d),'stations':d.station_id.nunique(),
                'mean_price':d.price.mean(),'mean_motorway':d.motorway.mean(),
                'mean_services':d.n_services.mean(),'mean_automated':d.automated.mean(),
                'regions':d.region_code.nunique()}
    csv(pd.DataFrame([describe(full,'full_complete_case'),describe(selected,label)]),
        f'{label}_sample_comparison.csv',tables)
    by_region=pd.concat([
        full.groupby('region_name').size().rename('full_station_months'),
        selected.groupby('region_name').size().rename('selected_station_months')],axis=1).fillna(0).reset_index()
    csv(by_region,f'{label}_region_composition.csv',tables)


def model_checks(monthly,weekly,meta,cfg,tables,large=False):
    full=regression_complete_cases(monthly)
    csv(pd.DataFrame([{'monthly_observations':len(monthly),'complete_case_observations':len(full),
         'excluded_missing_any_model_variable':len(monthly)-len(full),
         'stations_full':int(monthly.station_id.nunique()),'stations_complete_case':int(full.station_id.nunique())}]),
        'model_complete_case_audit.csv',tables)
    cap_list=[100000,250000]+([400000,0] if large else [])
    records=[]
    for cap in cap_list:
        label=f'cap_{cap}' if cap else 'full_sample'
        d,sub=select_station_sample(full,cap,cfg['random_seed'])
        print(f'  Regression sample {label}: {len(d):,} station-months / {d.station_id.nunique():,} stations',flush=True)
        sample_comparison(full,d,label,tables)
        r=fit_region_model(d,label,tables)
        records.append(r[r.term.isin(['motorway','n_services','automated'])])
    # Existing weekly panel contains only updates fresh under original 28-day cutoff.
    # Therefore a stricter 14-day sensitivity can be computed without new downloads.
    if cfg['max_price_age_days']>=14:
        w14=weekly[weekly.price_age_days.between(0,14)].copy()
        m14=join_monthly(monthly_from_weekly(w14,cfg),meta)
        del w14
        d14=regression_complete_cases(m14)
        sample14,_=select_station_sample(d14,250000,cfg['random_seed'])
        print(f'  Freshness 14 days: {len(m14):,} station-months',flush=True)
        r14=fit_region_model(sample14,'freshness_14d',tables)
        records.append(r14[r14.term.isin(['motorway','n_services','automated'])])
        csv(pd.DataFrame([{'threshold_days':14,'station_months':len(m14),
                           'stations':m14.station_id.nunique(),
                           'complete_case_station_months':len(d14)}]),
            'freshness_14d_panel_audit.csv',tables)
    out=pd.concat(records,ignore_index=True)
    csv(out,'robustness_main_coefficients.csv',tables)
    return out


def rebuild_56(cfg,meta,tables):
    """Optional costly reconstruction from the existing Parquet files, without overwriting base panels."""
    events=[]
    for y in range(cfg['analysis_start_year']-1,cfg['analysis_end_year']+1):
        raw=pipeline.load_year(y,cfg)
        clean,_=pipeline.audit_and_clean(y,raw,cfg,tables)
        events.append(clean)
        del raw
    e=pd.concat(events,ignore_index=True).drop_duplicates(['station_id','timestamp','price'])
    conflict=e.groupby(['station_id','timestamp'],sort=False).price.transform('nunique')>1
    e=e.loc[~conflict].copy()
    station_meta=[]
    for y in range(cfg['analysis_start_year'],cfg['analysis_end_year']+1):
        m,_=pipeline.metadata_for_year(y,tables)
        station_meta.append(m)
    cfg56={**cfg,'max_price_age_days':56}
    print('  Rebuilding 56-day panel from existing historical Parquet (may take several minutes)...',flush=True)
    weekly56,_=pipeline.construct_weekly_panel(
        e,pd.concat(station_meta,ignore_index=True),cfg56,tables,
        save_path=ROOT/'data'/'processed'/'diesel_weekly_panel_56d.parquet',
        audit_path=tables/'panel_audit_56d.csv')
    del e
    return weekly56


def report(monthly,cov,regions,spread,robust,cfg,tables,root,with56):
    base=pd.read_csv(ROOT/'outputs'/'tables'/'regression_main_terms.csv') if (ROOT/'outputs'/'tables'/'regression_main_terms.csv').exists() else pd.DataFrame()
    audits=pd.read_csv(ROOT/'outputs'/'tables'/'annual_data_quality.csv') if (ROOT/'outputs'/'tables'/'annual_data_quality.csv').exists() else pd.DataFrame()
    fmt=lambda x:f'{x:,.0f}'
    s=[
        '# Diesel Prices in France — Extended Econometric Analysis',
        '', '## Research question',
        '**How do diesel prices vary across French gas stations and over time, and how are these differences associated with geographical location and station characteristics?**',
        '', '## 1. Data and observation unit',
        f'The intended period is **{cfg["analysis_start_year"]}–{cfg["analysis_end_year"]}**. Historical 2020 prices initialize the beginning of 2021. The raw XML archives record **price updates**, not a daily census of prices.',
        f'Weekly station prices are the last reported diesel price at Sunday end-of-day, with a maximum age of {cfg["max_price_age_days"]} days. A station-month is the mean of at least {cfg["min_weekly_observations_per_month"]} observed weekly snapshots.',
        f'The existing constructed monthly dataset contains **{fmt(len(monthly))} station-month observations** from **{fmt(monthly.station_id.nunique())} stations**.',
        '', '## 2. Data quality and sample coverage',
        'Audit figures are counts of reported update rows. Reasons for invalidity can overlap; do not sum individual invalidity columns as if mutually exclusive.',
        '', '| Year | Gazole updates | Valid unique updates |', '|---|---:|---:|'
    ]
    for _,r in audits.iterrows():
        s.append(f'| {int(r.year)} | {fmt(r.gazole_rows)} | {fmt(r.valid_unique_updates)} |')
    s+=['','The station-year metadata and coverage diagnostics are in `outputs/analysis/tables/`. Metadata can change from year to year; the original parser does not explicitly reconstruct closures or shortages.','',
        '### Panel coverage', '', '| Year | Station-months | Stations | Unmapped region rows |', '|---|---:|---:|---:|']
    for _,r in cov.iterrows():
        s.append(f'| {int(r.year)} | {fmt(r.station_months)} | {fmt(r.stations)} | {fmt(r.unmapped_station_months)} |')
    s+=['','## 3. Geography: important methodological distinction',
        'The **region codes and region names** follow the French administrative region classification. However, each station is assigned a region **from its postal code**, not from a verified INSEE commune code. This is a **postal-code-derived regional proxy** and should not be described as an official station-to-region match.',
        'Corsican postcodes are assigned to the Corse region but not reliably separated into 2A/2B; some special overseas postal codes remain unmapped. Verify any ambiguous station against a trusted INSEE municipality mapping before making strong geographical claims.',
        '', '## 4. Descriptive geographical differences',
        'Regional price levels and within-region station dispersion are reported in the new CSV tables and figures. The month-adjusted region gap compares each region’s monthly average to the average across observed stations in that month; it is descriptive, not causal.',
        '', '## 5. Econometric specification',
        'The main extended model estimates **station-month diesel price** as a function of motorway status, service count, automated payment, **postal-derived region controls**, and **calendar-month fixed effects**. Inference uses standard errors clustered by station.',
        'Month fixed effects control common monthly movements; they do not make station characteristics exogenous. Region controls are different from the earlier two-character postal-prefix controls.',
        '', '## 6. Sensitivity checks',
        'The table below reports the key coefficients under different station-level sample caps and freshness assumptions. Values are EUR per litre. Subsamples are deterministic at station level, but estimates from different caps need not be identical.',
        '', '| Specification | Variable | Coefficient | 95% CI | p-value | Station-months |',
        '|---|---|---:|---|---:|---:|']
    for _,r in robust.iterrows():
        s.append(f'| {r.specification} | {r.term} | {r.coef_eur_per_litre:+.4f} | [{r.ci_lower_95:+.4f}, {r.ci_upper_95:+.4f}] | {r.p_value:.3g} | {fmt(r.station_months)} |')
    if len(base):
        old=base[(base.model=='M4 + station characteristics') & base.term.isin(['motorway','n_services','automated'])]
        s+=['','### Original model M4 (postal-prefix controls, original 250k cap)','',
            '| Variable | Original coefficient | 95% CI |', '|---|---:|---|']
        for _,r in old.iterrows():
            s.append(f'| {r.term} | {r.coef:+.4f} | [{r.ci_lower_95:+.4f}, {r.ci_upper_95:+.4f}] |')
    s+=['','**Freshness robustness:** 14 days is calculated from the existing 28-day weekly panel. '+
        ('The optional 56-day reconstruction was also run.' if with56 else 'A 56-day analysis was **not** run; use `--rebuild-56` for it.'),
        '**Sampling robustness:** by default the code compares 100k and 250k station-month caps. Use `--large-sample` for 400k and all complete-case observations (requires more RAM).',
        '', '## 7. Interpretation and limitations',
        '- A coefficient of +0.10 EUR/L means a conditional difference of **10 cents per litre**, not 10%.',
        '- Statistical significance is not the same as economic importance; compare coefficient magnitudes and confidence intervals.',
        '- OLS coefficients describe **conditional associations**, not causal impacts of motorway location, automation, war, or fuel subsidies.',
        '- A price update can be carried forward despite closure/shortage; the freshness rule reduces but cannot eliminate that risk.',
        '- Clustered errors handle repeated observations within station but not arbitrary nationwide or regional cross-station dependence.',
        '- The original automated-payment field defaults to False if the XML element is absent: zero may mean unreported, not necessarily no automation.',
        '- Regions are assigned from postal codes; cross-border postal exceptions and special territories need checking.',
        '- Missing covariates, incomplete reporting and station-level subsampling can affect external validity.',
        '- Different years may have different station composition; annual observed means are not volume-weighted national prices.',
        '', '## 8. Final tasks before submission',
        '1. Review geographical coverage and validate ambiguous postal codes against official INSEE reference data.',
        '2. Review price-update freshness, reporting gaps and station metadata coding.',
        '3. Compare sensitivity estimates, especially the motorway coefficient, rather than choosing only a favorable model.',
        '4. Discuss descriptive year trends and within-region dispersion alongside regression estimates.',
        '5. Write the final narrative in English with actual numbers and figure references; include a noncausal limitations paragraph.',
        '', '## Files',
        '- `outputs/analysis/figures/07_region_month_adjusted_differences.png`',
        '- `outputs/analysis/figures/08_within_region_station_dispersion.png`',
        '- `outputs/analysis/figures/09_region_year_heatmap.png`',
        '- `outputs/analysis/tables/robustness_main_coefficients.csv`',
        '- `outputs/analysis/tables/geography_assignment_audit.csv`',
        '- `outputs/analysis/tables/station_characteristics_stability.csv`',
        '- `outputs/analysis/tables/within_region_monthly_dispersion.csv`',
        '', 'Results require review of the underlying data and model assumptions before submission.'
    ]
    (root/'REPORT_ANALYSIS.md').write_text('\n'.join(s),encoding='utf-8')


def run(large=False,rebuild56=False):
    cfg=pipeline.get_config()
    p=ROOT/'data'/'processed'
    for f in ['diesel_weekly_panel.parquet','diesel_monthly_panel.parquet']:
        if not (p/f).exists():
            raise FileNotFoundError(f'Missing {p/f}. Run python run_all.py first.')
    out=ROOT/'outputs'/'analysis'; tables=out/'tables'; figures=out/'figures'
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    print('[Analysis 1/5] Checking annual station metadata and postal-derived regions...',flush=True)
    meta=station_geography(cfg,tables)
    print('[Analysis 2/5] Loading existing weekly and monthly panels...',flush=True)
    weekly=pd.read_parquet(p/'diesel_weekly_panel.parquet')
    monthly=join_monthly(pd.read_parquet(p/'diesel_monthly_panel.parquet'),meta)
    print('[Analysis 3/5] Generating geographical coverage, within-region variation and figures...',flush=True)
    cov,reg,spread,adjusted=descriptive(monthly,tables,figures)
    print('[Analysis 4/5] Testing 14/28-day freshness and regression sample caps...',flush=True)
    robust=model_checks(monthly,weekly,meta,cfg,tables,large=large)
    del weekly
    if rebuild56:
        w56=rebuild_56(cfg,meta,tables)
        m56=join_monthly(monthly_from_weekly(w56,cfg),meta)
        del w56
        d56=regression_complete_cases(m56)
        sample56,_=select_station_sample(d56,250000,cfg['random_seed'])
        r56=fit_region_model(sample56,'freshness_56d',tables)
        robust=pd.concat([robust,r56[r56.term.isin(['motorway','n_services','automated'])]],ignore_index=True)
        csv(robust,'robustness_main_coefficients.csv',tables)
        csv(pd.DataFrame([{'threshold_days':56,'station_months':len(m56),
            'stations':m56.station_id.nunique(),'complete_case_station_months':len(d56)}]),
            'freshness_56d_panel_audit.csv',tables)
    print('[Analysis 5/5] Writing analysis report...',flush=True)
    report(monthly,cov,reg,spread,robust,cfg,tables,out,rebuild56)
    print('SUCCESS. Open outputs/analysis/REPORT_ANALYSIS.md',flush=True)
    return out
