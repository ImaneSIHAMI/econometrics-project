"""France administrative region assignment from five-digit station postal codes.

The *region codes and names* correspond to French administrative regions.
The assignment from a station postal code is an approximation, NOT a verified
INSEE municipality-code join. Corsica cannot reliably be split into 2A/2B
from postal code alone, but the region is identifiable. Some overseas/special
postal codes are intentionally left unassigned.
"""
from __future__ import annotations
import pandas as pd

REGIONS = {
 '11':'Île-de-France', '24':'Centre-Val de Loire', '27':'Bourgogne-Franche-Comté',
 '28':'Normandie', '32':'Hauts-de-France', '44':'Grand Est', '52':'Pays de la Loire',
 '53':'Bretagne', '75':'Nouvelle-Aquitaine', '76':'Occitanie',
 '84':'Auvergne-Rhône-Alpes', '93':"Provence-Alpes-Côte d’Azur", '94':'Corse',
 '01':'Guadeloupe', '02':'Martinique', '03':'Guyane', '04':'La Réunion', '06':'Mayotte'
}
DEPT_TO_REGION = {}
for region, departments in {
 '11':['75','77','78','91','92','93','94','95'],
 '24':['18','28','36','37','41','45'],
 '27':['21','25','39','58','70','71','89','90'],
 '28':['14','27','50','61','76'],
 '32':['02','59','60','62','80'],
 '44':['08','10','51','52','54','55','57','67','68','88'],
 '52':['44','49','53','72','85'],
 '53':['22','29','35','56'],
 '75':['16','17','19','23','24','33','40','47','64','79','86','87'],
 '76':['09','11','12','30','31','32','34','46','48','65','66','81','82'],
 '84':['01','03','07','15','26','38','42','43','63','69','73','74'],
 '93':['04','05','06','13','83','84'],
 '94':['20']
}.items():
    for dep in departments:
        if dep in DEPT_TO_REGION:
            raise ValueError(f'Department duplicated: {dep}')
        DEPT_TO_REGION[dep]=region

OVERSEAS={'971':'01','972':'02','973':'03','974':'04','976':'06'}

def classify_postal_code(code):
    """Return (department proxy, region code, region name, assignment status)."""
    if code is None or pd.isna(code):
        return (None,None,None,'missing_postal_code')
    cp=str(code).strip()
    if cp.endswith('.0') and cp[:-2].isdigit(): cp=cp[:-2]
    if not cp.isdigit() or len(cp)>5 or len(cp)<4:
        return (None,None,None,'invalid_postal_code')
    cp=cp.zfill(5)
    if cp.startswith('97'):
        dept=cp[:3]
        reg=OVERSEAS.get(dept)
        if reg: return (dept,reg,REGIONS[reg],'postal_proxy_overseas')
        return (dept,None,None,'unmapped_overseas_postal_code')
    if cp.startswith('98'):
        return (cp[:3],None,None,'special_postal_code_unmapped')
    dep=cp[:2]
    reg=DEPT_TO_REGION.get(dep)
    if reg is None:
        return (dep,None,None,'unmapped_postal_code')
    if dep=='20':
        return ('20 (Corse)',reg,REGIONS[reg],'corse_region_only')
    return (dep,reg,REGIONS[reg],'postal_proxy_mainland')


def attach_geography(df, postal_col='postal_code'):
    out=df.copy()
    unique=out[postal_col].astype('string').drop_duplicates()
    mapping=pd.DataFrame([
        (p,*classify_postal_code(p)) for p in unique
    ],columns=[postal_col,'department_proxy','region_code','region_name','geography_status'])
    # map via string to preserve NA safely
    out[postal_col]=out[postal_col].astype('string')
    return out.merge(mapping,on=postal_col,how='left',validate='many_to_one')
