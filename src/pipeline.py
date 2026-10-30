"""Validated diesel panel, descriptive outputs and econometric models.

Run from the repository root using `python run_all.py`.
Raw and interim files are NEVER changed by this module.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')


def get_config():
    return json.loads((ROOT/'config.json').read_text(encoding='utf-8'))


def load_year(year: int, cfg: dict):
    path = ROOT/'data'/'interim'/f'prices_{year}.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run python src/download_data.py then python src/parse_data.py')
    # Arrow reads only Gazole rows and required columns, avoiding all-fuel memory use.
    cols = ['station_id','fuel','timestamp','price','source_year']
    tab = pq.read_table(path, columns=cols, filters=[('fuel', '==', cfg['fuel'])])
    df = tab.to_pandas()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    # Confirmed from the user's historical Parquet samples:
    # 2020: median 1252 -> EUR 1.252/L; 2021: median 1444 -> EUR 1.444/L.
    # 2022 onwards: prices already in EUR/L (2022 median 1.862).
    # Convert ONLY the historical years, BEFORE the price-validity audit.
    if year in (2020, 2021):
        df['price'] = df['price'] / 1000.0
    return df


def audit_and_clean(year, df, cfg, tables):
    year_start = pd.Timestamp(year, 1, 1)
    year_end = pd.Timestamp(year+1, 1, 1)
    mask_date = df.timestamp.ge(year_start) & df.timestamp.lt(year_end)
    valid = df.station_id.notna() & mask_date & df.price.between(cfg['min_price_eur'],cfg['max_price_eur'])
    good = df.loc[valid, ['station_id','timestamp','price']].copy()
    # Same station and timestamp with different prices: ambiguous, exclude ALL conflicting rows.
    conflict_count = good.groupby(['station_id','timestamp'],sort=False)['price'].transform('nunique') > 1
    conflicting_keys = good.loc[conflict_count,['station_id','timestamp']].drop_duplicates()
    good = good.loc[~conflict_count].drop_duplicates(['station_id','timestamp','price'])
    good = good.sort_values(['timestamp','station_id']).reset_index(drop=True)
    audit = {
        'year':year,'gazole_rows':len(df),'missing_station_id':int(df.station_id.isna().sum()),
        'missing_timestamp':int(df.timestamp.isna().sum()),'missing_price':int(df.price.isna().sum()),
        'timestamp_outside_archive_year':int((df.timestamp.notna() & ~mask_date).sum()),
        'implausible_or_nonpositive_price':int((df.price.notna() & ~df.price.between(cfg['min_price_eur'],cfg['max_price_eur'])).sum()),
        'conflicting_station_timestamp_keys':len(conflicting_keys),
        'valid_unique_updates':len(good),'stations_with_valid_updates':good.station_id.nunique()
    }
    write_csv(conflicting_keys.assign(year=year),tables/f'conflicts_{year}.csv')
    return good, audit


def metadata_for_year(year, tables):
    p=ROOT/'data'/'interim'/f'stations_{year}.parquet'
    if not p.exists():
        raise FileNotFoundError(f'Missing {p}')
    m=pd.read_parquet(p)
    expected=['station_id','postal_code','road_type','city','latitude','longitude','automate_24_24','n_services']
    for col in expected:
        if col not in m.columns: m[col]=pd.NA
    m=m[expected].copy()
    m=m[m.station_id.notna()]
    # Flag metadata inconsistency rather than silently assuming the first row is correct.
    count=int(m.station_id.duplicated(keep=False).sum())
    if count:
        write_csv(m[m.station_id.duplicated(keep=False)],tables/f'metadata_duplicates_{year}.csv')
    m=m.drop_duplicates('station_id',keep='first').copy()
    m['year']=year
    m['motorway']=m['road_type'].astype('string').str.upper().map({'A':1.0,'R':0.0})
    m['n_services']=pd.to_numeric(m.n_services,errors='coerce')
    m['automated']=m.automate_24_24.map({True:1.0,False:0.0})
    # First two postal-code characters are not reliable for all overseas territories.
    m['department_code']=m.postal_code.astype('string').str[:2]
    return m, {'year':year,'station_metadata_rows':len(m),'duplicate_station_rows_before_choice':count,'unknown_road_type':int(m.motorway.isna().sum())}


def construct_weekly_panel(events, meta, cfg, tables):
    start=pd.Timestamp(cfg['analysis_start_year'],1,1)
    end=pd.Timestamp(cfg['analysis_end_year'],12,31,23,59,59)
    # Sunday end-of-day: observed updates up to the cutoff, never future updates.
    dates=pd.date_range(start.normalize(),end.normalize(),freq='W-SUN') + pd.Timedelta(hours=23,minutes=59,seconds=59)
    # Restrict candidate stations to those with a valid update in the analysis years.
    candidate=events.loc[events.timestamp.ge(start),'station_id'].dropna().unique()
    if not len(candidate): raise ValueError('No valid diesel updates during the analysis period')
    weeks=pd.DataFrame({'timestamp':np.tile(dates.to_numpy(),len(candidate)),
                        'station_id':np.repeat(candidate,len(dates))})
    # merge_asof requires global sorting by timestamp, NOT by station then timestamp.
    weeks=weeks.sort_values('timestamp',kind='mergesort').reset_index(drop=True)
    events=events[['station_id','timestamp','price']].sort_values('timestamp',kind='mergesort').reset_index(drop=True)
    events=events.rename(columns={'timestamp':'last_update'})
    result=pd.merge_asof(weeks,events,left_on='timestamp',right_on='last_update',by='station_id',direction='backward',allow_exact_matches=True)
    result['price_age_days']=(result.timestamp-result.last_update).dt.total_seconds()/86400
    valid=result.price.notna() & result.price_age_days.between(0,cfg['max_price_age_days'])
    audit={'candidate_stations':int(len(candidate)), 'weekly_grid_rows':int(len(result)),
           'weekly_price_known_before_freshness_filter':int(result.price.notna().sum()),
           'weekly_price_valid_after_freshness_filter':int(valid.sum()),
           'weekly_rows_rejected_stale_or_missing':int((~valid).sum()),
           'weekly_grid_start':str(dates.min()),'weekly_grid_end':str(dates.max())}
    result=result.loc[valid].copy()
    result['week']=result.timestamp.dt.normalize()
    result['year']=result.week.dt.year
    result['month']=result.week.dt.to_period('M').astype(str)
    result=result.merge(meta,on=['station_id','year'],how='left',validate='many_to_one')
    audit['weekly_rows_missing_matching_year_metadata']=int(result.road_type.isna().sum())
    audit['stations_in_valid_weekly_panel']=int(result.station_id.nunique())
    result=result.drop(columns=['timestamp'])
    (ROOT/'data'/'processed').mkdir(parents=True, exist_ok=True)
    result.to_parquet(ROOT/'data'/'processed'/'diesel_weekly_panel.parquet',index=False)
    write_csv(pd.DataFrame([audit]),tables/'panel_audit.csv')
    return result, audit


def make_monthly(weekly,cfg):
    keys=['station_id','month']
    monthly=(weekly.groupby(keys,sort=False,observed=True)
             .agg(price=('price','mean'),weeks_observed=('price','size'),
                  year=('year','first'),motorway=('motorway','first'),
                  n_services=('n_services','first'),automated=('automated','first'),
                  department_code=('department_code','first'),
                  latitude=('latitude','first'),longitude=('longitude','first'))
             .reset_index())
    monthly=monthly[monthly.weeks_observed>=cfg['min_weekly_observations_per_month']].copy()
    (ROOT/'data'/'processed').mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(ROOT/'data'/'processed'/'diesel_monthly_panel.parquet',index=False)
    return monthly


def describe(weekly,monthly,tables,figures):
    by_month=(monthly.groupby('month',sort=True)
              .agg(mean_price=('price','mean'),median_price=('price','median'),
                   sd_price=('price','std'),n_station_months=('price','size'),
                   n_stations=('station_id','nunique')).reset_index())
    write_csv(by_month,tables/'mean_price_by_month.csv')
    by_year=(monthly.groupby('year',sort=True)
             .agg(mean_price=('price','mean'),median_price=('price','median'),
                  sd_price=('price','std'),n_station_months=('price','size'),
                  n_stations=('station_id','nunique')).reset_index())
    write_csv(by_year,tables/'mean_price_by_year.csv')
    by_road=(monthly.dropna(subset=['motorway']).groupby('motorway')
             .agg(mean_price=('price','mean'),n=('price','size'),stations=('station_id','nunique')).reset_index())
    write_csv(by_road,tables/'price_by_road_type.csv')
    by_dept=(monthly.dropna(subset=['department_code']).groupby('department_code')
             .agg(mean_price=('price','mean'),n=('price','size'),stations=('station_id','nunique')).reset_index())
    write_csv(by_dept,tables/'price_by_department_postal_prefix.csv')
    by_station=monthly.groupby('station_id').agg(n_months=('month','nunique'),mean_price=('price','mean')).reset_index()
    write_csv(by_station,tables/'observations_per_station.csv')
    coverage=monthly.groupby('month').agg(stations=('station_id','nunique')).reset_index()
    write_csv(coverage,tables/'station_coverage_by_month.csv')
    plt.rcParams.update({'figure.dpi':120,'font.size':10})
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.plot(pd.to_datetime(by_month.month),by_month.mean_price,linewidth=1.7)
    ax.set(title='Mean diesel price by month (station-weighted)',ylabel='EUR per litre',xlabel='Month')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(figures/'01_monthly_price_trend.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.bar(by_year.year.astype(str),by_year.mean_price)
    ax.set(title='Mean diesel price by year',ylabel='EUR per litre',xlabel='Year')
    fig.tight_layout();fig.savefig(figures/'02_yearly_price_means.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.hist(monthly.price,bins=50,alpha=.85)
    ax.set(title='Distribution of station-month diesel prices',xlabel='EUR per litre',ylabel='Station-month observations')
    fig.tight_layout();fig.savefig(figures/'03_price_distribution.png');plt.close(fig)
    if len(by_road):
        fig,ax=plt.subplots(figsize=(7,4.5))
        ax.bar(by_road.motorway.map({0.0:'Non-motorway',1.0:'Motorway'}),by_road.mean_price)
        ax.set(title='Unadjusted diesel prices by road type',ylabel='EUR per litre')
        fig.tight_layout();fig.savefig(figures/'04_motorway_comparison.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(pd.to_datetime(coverage.month),coverage.stations)
    ax.set(title='Number of observed stations by month',ylabel='Stations',xlabel='Month')
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(figures/'05_station_coverage.png');plt.close(fig)
    if len(by_dept):
        top=by_dept.nlargest(15,'n').sort_values('mean_price')
        fig,ax=plt.subplots(figsize=(9,5))
        ax.barh(top.department_code.astype(str),top.mean_price)
        ax.set(title='Mean diesel prices: 15 best-covered postal prefixes',xlabel='EUR per litre',ylabel='Postal prefix (not official department)')
        fig.tight_layout();fig.savefig(figures/'06_postal_prefix_means.png');plt.close(fig)
    return by_year,by_month,by_road


def regression_data(monthly,cfg,tables):
    d=monthly.copy()
    # Compare identical observations across specifications; missing covariates excluded.
    d=d.dropna(subset=['price','motorway','n_services','automated','department_code','month','station_id']).copy()
    d=d[d.department_code.str.len()==2]
    # Limit number of columns from sparse regions by grouping rare postal prefixes.
    counts=d.department_code.value_counts()
    d['geo']=d.department_code.where(d.department_code.isin(counts[counts>=100].index),'OTHER')
    # Cap only if necessary, with a deterministic station-level subsample, not row-level.
    original=len(d)
    if len(d)>cfg['regression_sample_cap']:
        stations=d.station_id.drop_duplicates().sample(frac=1,random_state=cfg['random_seed']).to_numpy()
        selected=[]; total=0
        sizes=d.station_id.value_counts()
        for station in stations:
            if total+int(sizes[station])>cfg['regression_sample_cap']:
                continue
            selected.append(station);total+=int(sizes[station])
            if total>=cfg['regression_sample_cap']*.98:break
        d=d[d.station_id.isin(selected)].copy()
    audit={'full_station_months':int(len(monthly)),'complete_case_station_months':int(original),
           'regression_station_months':int(len(d)),'regression_stations':int(d.station_id.nunique()),
           'station_level_subsample_applied':bool(original>len(d)),
           'max_station_months':int(cfg['regression_sample_cap'])}
    write_csv(pd.DataFrame([audit]),tables/'regression_sample_audit.csv')
    return d,audit


def estimate_models(d,cfg,tables):
    if len(d)<cfg['min_regression_observations'] or d.station_id.nunique()<3:
        return pd.DataFrame(), ['Insufficient observations or stations for regressions.']
    # Countrywide month fixed effects are absorbed using Frisch-Waugh-Lovell.
    region_dummies=pd.get_dummies(d.geo.astype(str),prefix='geo',drop_first=True,dtype=float)
    xbase=d[['motorway','n_services','automated']].astype(float).reset_index(drop=True)
    d=d.reset_index(drop=True)
    regions=region_dummies.reset_index(drop=True)
    y=d.price.astype(float).reset_index(drop=True)
    groups=d.station_id.astype(str).to_numpy()
    outputs=[]; errors=[]
    models=[('M1 motorway only',['motorway'],False,False),
            ('M2 + geography',['motorway'],True,False),
            ('M3 + month FE',['motorway'],True,True),
            ('M4 + station characteristics',['motorway','n_services','automated'],True,True)]
    for name,cols,geo,month_fe in models:
        X=xbase[cols].copy()
        if geo:X=pd.concat([X,regions],axis=1)
        if month_fe:
            # Subtract each calendar month's cross-sectional mean from every variable.
            X=X-X.groupby(d.month).transform('mean')
            yreg=y-y.groupby(d.month).transform('mean')
            # no intercept: month FE have been absorbed
        else:
            X=sm.add_constant(X,has_constant='add');yreg=y
        # Drop columns with no variation after within-month transformation.
        X=X.loc[:,(X.std(axis=0).fillna(0)>1e-12) | (X.columns=='const')]
        try:
            fit=sm.OLS(yreg.to_numpy(),X.to_numpy()).fit(cov_type='cluster',cov_kwds={'groups':groups},use_t=False)
            # Overall R2 accounting for absorbed month fixed effects.
            rss=float(np.square(fit.resid).sum())
            tss=float(np.square(y-y.mean()).sum())
            r2=1-rss/tss if tss>0 else np.nan
            for j,col in enumerate(X.columns):
                outputs.append({'model':name,'term':col,'coef':float(fit.params[j]),
                                'se_cluster_station':float(fit.bse[j]),'z_stat':float(fit.tvalues[j]),
                                'p_value':float(fit.pvalues[j]),'ci_lower_95':float(fit.conf_int()[j,0]),
                                'ci_upper_95':float(fit.conf_int()[j,1]),
                                'n_station_months':len(d),'n_stations':d.station_id.nunique(),
                                'overall_r2':r2,'month_fixed_effects':month_fe,'geo_controls':geo})
            (tables/f'{name.split()[0]}_model_summary.txt').write_text(
                f'{name}\nStation-clustered covariance; same complete-case sample across models.\n'
                f'Month effects absorbed using FWL: {month_fe}. Overall R2: {r2:.4f}\n\n'
                +fit.summary(xname=list(X.columns)).as_text(),encoding='utf-8')
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    out=pd.DataFrame(outputs)
    if len(out):
        write_csv(out,tables/'regression_coefficients.csv')
        write_csv(out[out.term.isin(['motorway','n_services','automated','const'])],tables/'regression_main_terms.csv')
    return out,errors


def write_report(audits,meta_audits,panel_audit,reg_audit,by_year,by_month,by_road,reg,errors,cfg,outputs):
    def n(x):return f'{int(x):,}'
    text=[
      '# Fuel Price Differences in France — Simple Results Report',
      '', '**Status:** automatically generated from the local files. This report contains no invented regression estimates.',
      '', '## 1. Objective',
      'Study how French diesel (Gazole) prices vary across stations and months, and how these differences are associated with road type, geography and observable station characteristics.',
      '', '## 2. What is one observation?',
      'A **weekly** observation is the latest reported Gazole price at a station by Sunday 23:59:59, provided the price update is not older than '+str(cfg['max_price_age_days'])+' days. A **monthly** observation averages the valid weekly observations at that station in that calendar month.',
      'This is a constructed panel, **not** a direct daily survey. No price is filled using a future update. An unchanged price may remain valid longer than the freshness threshold, so coverage is conservative.',
      '', '## 3. Data audit',
      '| Year | Diesel raw updates | Valid unique updates | Stations | Conflicting keys |',
      '|---|---:|---:|---:|---:|'
    ]
    for a in audits:
        text.append(f"| {a['year']} | {n(a['gazole_rows'])} | {n(a['valid_unique_updates'])} | {n(a['stations_with_valid_updates'])} | {n(a['conflicting_station_timestamp_keys'])} |")
    text.extend(['','**Important:** the archive records price updates, not necessarily one observation per station per day. Ambiguous same-station/same-timestamp prices are excluded, not arbitrarily resolved.',
                 'The original archives and yearly Parquet files remain unchanged. See `outputs/tables/` for full audits.',
                 '', '## 4. Panel construction',
                 f"- Candidate stations: **{n(panel_audit['candidate_stations'])}**.",
                 f"- Station-week grid rows: **{n(panel_audit['weekly_grid_rows'])}**.",
                 f"- Valid station-week observations: **{n(panel_audit['weekly_price_valid_after_freshness_filter'])}**.",
                 f"- Station-week rows missing or too stale: **{n(panel_audit['weekly_rows_rejected_stale_or_missing'])}**.",
                 f"- Missing matching-year station metadata among retained weeks: **{n(panel_audit['weekly_rows_missing_matching_year_metadata'])}**.",
                 f"- Final station-month observations (before regression complete-case filtering): **{n(reg_audit['full_station_months'])}**.",
                 f"- Regression station-month observations: **{n(reg_audit['regression_station_months'])}** across **{n(reg_audit['regression_stations'])}** stations.",
                 f"- Station-level regression subsampling: **{'yes' if reg_audit['station_level_subsample_applied'] else 'no'}**.",
                 '', '## 5. Descriptive findings',
                 'Annual station-month weighted means (EUR per litre):',
                 '', '| Year | Mean | Stations | Station-months |', '|---|---:|---:|---:|'])
    for _,r in by_year.iterrows():
        text.append(f"| {int(r.year)} | {r.mean_price:.3f} | {n(r.n_stations)} | {n(r.n_station_months)} |")
    text.extend(['','**Interpretation:** these are observed station-month averages, not population-weighted national pump prices. Changes in which stations report may affect comparisons.',
                 '', '## 6. Regression results',
                 'All models use the same complete-case station-month sample, with station-clustered standard errors. M3/M4 absorb calendar-month effects using the Frisch–Waugh–Lovell transformation.',
                 'The geography variable is the first two characters of the postal code, **not an official region or universally reliable department identifier**. Do not describe these controls as regional fixed effects.',
                 ''])
    if len(reg):
        text.extend(['| Model | Term | Coefficient (EUR/L) | Cluster SE | p-value |', '|---|---|---:|---:|---:|'])
        for _,r in reg[reg.term.isin(['motorway','n_services','automated'])].iterrows():
            text.append(f"| {r['model']} | {r.term} | {r.coef:+.4f} | {r.se_cluster_station:.4f} | {r.p_value:.4g} |")
        text.extend(['','**Reading the coefficients:** the motorway coefficient is the adjusted difference in EUR/L between stations coded as motorway and non-motorway, conditional on the regressors included in that model. It is an association, not a causal effect.'])
    else:text.append('No regressions estimated; see diagnostics below.')
    if errors:text.extend(['','**Model warnings/errors:**']+['- '+e for e in errors])
    text.extend(['','## 7. What this project does NOT establish',
                 '- It does not identify the causal effect of a motorway location, services, a war, or a government policy.',
                 '- It does not establish whether missing or stale prices reflect station closures, shortages, or unchanged prices.',
                 '- It does not yet use a validated official region code or regional cost-of-living series.',
                 '- It does not estimate a station fixed-effects model; time-invariant station characteristics would be absorbed by station fixed effects.',
                 '- If station-level regression subsampling was necessary, inference describes the analyzed station sample; results should be checked against larger caps on capable computers.',
                 '', '## 8. Next checks before final submission',
                 '1. Inspect coverage and stale-price exclusions, especially around the beginning of 2021.',
                 '2. Repeat the panel construction with 14- and 56-day freshness limits and compare the results.',
                 '3. Confirm the historical XML road-type coding and automation/services fields.',
                 '4. Join a verified INSEE department/region reference if the final question is explicitly regional.',
                 '5. Discuss station-level sampling, reporting selection and repeated-observation dependence.',
                 '6. Use the exported tables and figures to write a careful final report and presentation.',
                 '', '## 9. Files to inspect',
                 '- `outputs/figures/01_monthly_price_trend.png` — price evolution.',
                 '- `outputs/figures/05_station_coverage.png` — sample composition over time.',
                 '- `outputs/tables/regression_main_terms.csv` — coefficients and standard errors.',
                 '- `outputs/tables/price_by_department_postal_prefix.csv` — provisional geographical summary.',
                 '- `data/processed/diesel_weekly_panel.parquet` and `diesel_monthly_panel.parquet` — constructed datasets.',
                 '', '## 10. Important caveat about historical archive XML',
                 'The parser provided in the original GitHub repository extracts `<prix>` updates and annual station metadata but does not reconstruct explicit closure/reopening intervals or shortage periods. Therefore the price panel is a **reported-price proxy** rather than a verified record of a price actually available to purchase on every Sunday. Do not make claims about actual availability without extending the parser.',
                 ''])
    report='\n'.join(text)
    (outputs/'REPORT_SIMPLE.md').write_text(report,encoding='utf-8')
    return report


def run():
    cfg=get_config()
    if cfg['analysis_start_year']>cfg['analysis_end_year']:raise ValueError('start_year must be <= end_year')
    out=ROOT/'outputs';tables=out/'tables';figures=out/'figures'
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    print('[1/7] Auditing annual Gazole updates (2020 initialization + study years)...',flush=True)
    events=[];audits=[];meta=[];meta_audits=[]
    for year in range(cfg['analysis_start_year']-1,cfg['analysis_end_year']+1):
        raw=load_year(year,cfg)
        cleaned,audit=audit_and_clean(year,raw,cfg,tables)
        events.append(cleaned);audits.append(audit)
        if year>=cfg['analysis_start_year']:
            m,ma=metadata_for_year(year,tables)
            meta.append(m);meta_audits.append(ma)
        print(f"  {year}: {audit['gazole_rows']:,} raw updates, {audit['valid_unique_updates']:,} valid unique updates",flush=True)
        del raw
    write_csv(pd.DataFrame(audits),tables/'annual_data_quality.csv')
    write_csv(pd.DataFrame(meta_audits),tables/'annual_station_metadata_quality.csv')
    print('[2/7] Combining valid historical updates...',flush=True)
    events=pd.concat(events,ignore_index=True)
    # Events may be repeated in successive annual archive files: prefer unambiguous event keys.
    events=events.drop_duplicates(['station_id','timestamp','price'])
    global_conflict=events.groupby(['station_id','timestamp'],sort=False).price.transform('nunique')>1
    if global_conflict.any():
        write_csv(events.loc[global_conflict],tables/'cross_archive_conflicts.csv')
        events=events.loc[~global_conflict].copy()
    meta=pd.concat(meta,ignore_index=True)
    print('[3/7] Building end-of-week diesel panel...',flush=True)
    weekly,panel_audit=construct_weekly_panel(events,meta,cfg,tables)
    del events
    print('[4/7] Aggregating to station-month panel...',flush=True)
    monthly=make_monthly(weekly,cfg)
    print('[5/7] Producing descriptive tables and figures...',flush=True)
    by_year,by_month,by_road=describe(weekly,monthly,tables,figures)
    del weekly
    print('[6/7] Estimating station-clustered OLS models...',flush=True)
    d,reg_audit=regression_data(monthly,cfg,tables)
    reg,errors=estimate_models(d,cfg,tables)
    print('[7/7] Writing the simple English results report...',flush=True)
    write_report(audits,meta_audits,panel_audit,reg_audit,by_year,by_month,by_road,reg,errors,cfg,out)
    print('\nSUCCESS. Open outputs/REPORT_SIMPLE.md and outputs/figures/.',flush=True)
    return {'audits':audits,'panel':panel_audit,'regression':reg_audit,'model_errors':errors}
