"""New explicit discrete-event reference model, not a claim of ALP equivalence."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from analyze import iqr_mask, samples, valid_values
from prepare import ROOT, read_csv, save_csv, save_json


@dataclass(frozen=True)
class Party:
    id: str
    arrival: float
    size: int
    kind: str
    kiosk: bool
    dine_in: bool
    kiosk_seconds: float
    food_seconds: float
    dining_seconds: float
    group_size_imputed: bool = False


def run(parties, *, kiosks=2, seats=35, horizon=3600, patience=None, log_events=False):
    if isinstance(kiosks,bool) or isinstance(seats,bool) or int(kiosks)!=kiosks or int(seats)!=seats or kiosks<1 or seats<1:
        raise ValueError('Capacities must be positive integers')
    if not math.isfinite(horizon) or horizon<=0: raise ValueError('Invalid horizon')
    if patience is not None and (not math.isfinite(patience) or patience<0): raise ValueError('Invalid patience')
    if len({p.id for p in parties})!=len(parties): raise ValueError('Duplicate party IDs')
    for p in parties:
        if isinstance(p.size,bool) or int(p.size)!=p.size or p.size<1 or (p.dine_in and p.size>seats):
            raise ValueError('Group size must be positive and fit available seating')
        if p.kind not in ['customer','courier','walkaway']: raise ValueError('Unknown party kind')
        if not all(math.isfinite(v) and v>=0 for v in [p.arrival,p.kiosk_seconds,p.food_seconds,p.dining_seconds]):
            raise ValueError('Invalid time input')
    # Arrival at H is outside [0,H); existing parties are drained after H to avoid discarding censored records.
    included=[p for p in parties if p.arrival<horizon]
    lookup={p.id:p for p in included};serial=itertools.count();events=[]
    state={};kiosk_q=deque();seat_q=deque();busy_kiosks=occupied=0
    peak_kiosks=peak_seats=0;last=0.;kiosk_area=seat_area=0.;event_log=[]
    priority={'dining_done':0,'kiosk_done':0,'food_ready':0,'timeout':1,'arrival':2}

    def schedule(t,kind,pid): heapq.heappush(events,(t,priority[kind],next(serial),kind,pid))
    def finish(pid,t,reason):
        state[pid].update(status='departed',departure=t,departure_reason=reason)
    def start_kiosks(t):
        nonlocal busy_kiosks,peak_kiosks
        while kiosk_q and busy_kiosks<kiosks:
            pid=kiosk_q.popleft();p=lookup[pid];r=state[pid]
            r.update(status='ordering',kiosk_start=t,kiosk_wait=t-p.arrival)
            busy_kiosks+=1;peak_kiosks=max(peak_kiosks,busy_kiosks)
            schedule(t+p.kiosk_seconds,'kiosk_done',pid)
    def start_seats(t):
        nonlocal occupied,peak_seats
        while seat_q:
            pid=seat_q[0];p=lookup[pid];r=state[pid]
            if r['status']!='seat_queue': seat_q.popleft();continue
            if occupied+p.size>seats: break  # Strict FIFO, no bypass by smaller groups.
            seat_q.popleft();occupied+=p.size;peak_seats=max(peak_seats,occupied)
            r.update(status='dining',seat_start=t,seat_wait=t-r['food_ready'])
            schedule(t+p.dining_seconds,'dining_done',pid)

    for p in included: schedule(p.arrival,'arrival',p.id)
    while events:
        t,_,_,kind,pid=heapq.heappop(events);p=lookup[pid]
        elapsed=max(0,min(t,horizon)-min(last,horizon))
        kiosk_area+=busy_kiosks*elapsed;seat_area+=occupied*elapsed;last=t
        if kind=='arrival':
            state[pid]={'party_id':pid,'arrival':p.arrival,'size':p.size,'kind':p.kind,
                'kiosk':p.kiosk,'dine_in':p.dine_in,'group_size_imputed':p.group_size_imputed,
                'kiosk_seconds':p.kiosk_seconds,'food_seconds':p.food_seconds,'dining_seconds':p.dining_seconds,
                'status':'food_wait','kiosk_start':None,'kiosk_wait':0.,'food_ready':None,
                'seat_start':None,'seat_wait':None,'departure':None,'departure_reason':None}
            if p.kind=='walkaway': finish(pid,t,'observed_walkaway')
            elif p.kiosk:
                state[pid]['status']='kiosk_queue';kiosk_q.append(pid);start_kiosks(t)
            else: schedule(t+p.food_seconds,'food_ready',pid)
        elif kind=='kiosk_done':
            busy_kiosks-=1;state[pid]['status']='food_wait'
            schedule(t+p.food_seconds,'food_ready',pid);start_kiosks(t)
        elif kind=='food_ready':
            r=state[pid];r['food_ready']=t
            if not p.dine_in: finish(pid,t,'pickup')
            else:
                r['status']='seat_queue';seat_q.append(pid)
                if patience is not None: schedule(t+patience,'timeout',pid)
                start_seats(t)
        elif kind=='dining_done':
            occupied-=p.size;finish(pid,t,'dined');start_seats(t)
        elif kind=='timeout':
            r=state[pid]
            if r['status']=='seat_queue':
                r['seat_wait']=t-r['food_ready'];finish(pid,t,'seat_timeout_takeaway');start_seats(t)
        if not 0<=busy_kiosks<=kiosks or not 0<=occupied<=seats: raise AssertionError('Resource invariant failed')
        if log_events:
            event_log.append({'time_seconds':t,'event':kind,'party_id':pid,'busy_kiosks':busy_kiosks,
                'occupied_seats':occupied,'kiosk_queue_groups':len(kiosk_q),
                'seat_queue_groups':sum(state[j]['status']=='seat_queue' for j in seat_q),
                'departed_groups':sum(r['departure'] is not None for r in state.values()),'arrived_groups':len(state)})
    if busy_kiosks or occupied or any(r['departure'] is None for r in state.values()):
        raise AssertionError('Failed to drain all arrivals')
    records=list(state.values())
    for r in records:
        r['completed_by_horizon']=r['departure']<=horizon
        r['time_in_system']=r['departure']-r['arrival']
        r['restricted_time_in_system']=min(r['departure'],horizon)-r['arrival']
        r['food_ready_elapsed']=r['food_ready']-r['arrival'] if r['food_ready'] is not None else None
    departed=[r for r in records if r['completed_by_horizon']]
    customers=[r for r in records if r['kind']=='customer']
    def mean(rows,key):
        values=[r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None
    result={'arrived_groups':len(records),'departed_groups_by_horizon':len(departed),
        'remaining_groups_at_horizon':len(records)-len(departed),
        'arrived_people':sum(r['size'] for r in records),
        'departed_people_by_horizon':sum(r['size'] for r in departed),
        'remaining_people_at_horizon':sum(r['size'] for r in records if not r['completed_by_horizon']),
        'customer_groups':len(customers),
        'customer_restricted_mean_seconds':mean(customers,'restricted_time_in_system'),
        'customer_full_cohort_mean_seconds':mean(customers,'time_in_system'),
        'customer_food_ready_mean_seconds':mean(customers,'food_ready_elapsed'),
        'customer_kiosk_wait_mean_seconds':mean([r for r in customers if r['kiosk']],'kiosk_wait'),
        'customer_seat_wait_mean_seconds':mean([r for r in customers if r['dine_in']],'seat_wait'),
        'customer_completed_only_mean_seconds':mean([r for r in customers if r['completed_by_horizon']],'time_in_system'),
        'kiosk_utilization':kiosk_area/(kiosks*horizon),'seat_utilization':seat_area/(seats*horizon),
        'peak_busy_kiosks':peak_kiosks,'peak_occupied_seats':peak_seats,
        'seat_timeout_groups':sum(r['departure_reason']=='seat_timeout_takeaway' for r in records),
        'final_departure_seconds':max((r['departure'] for r in records),default=0),
        'imputed_group_count':sum(r['group_size_imputed'] for r in records)}
    return result,records,event_log


def draw_parties(data_dir,config,seed):
    grouped=samples(data_dir)
    pools={}
    for key in ['kiosk_dataset','food_delay_dataset','dining_dataset']:
        x=valid_values(grouped[config[key]])
        if config['iqr_filter']: x=x[iqr_mask(x)[0]]
        pools[key]=np.sort(x)
    for key in ['unknown_kiosk_probability','unknown_dine_in_probability']:
        if not 0<=config[key]<=1: raise ValueError('Invalid probability')
    # Five uniforms per group, including unused fields: scenarios keep their common random numbers.
    rng=np.random.default_rng(seed);parties=[]
    for row in read_csv(Path(data_dir)/'arrivals.csv'):
        u=rng.random(5);kind=row['kind']
        kiosk=kind=='customer' and (row['kiosk_observed']=='1' or u[0]<config['unknown_kiosk_probability'])
        dine=kind=='customer' and (row['dine_in_observed']=='1' if row['dine_in_observed'] else u[1]<config['unknown_dine_in_probability'])
        def sample(key,j):
            x=pools[key];return float(x[min(int(u[j]*len(x)),len(x)-1)])
        size=int(float(row['group_size'])) if row['group_size'] else config['missing_group_size']
        parties.append(Party(row['arrival_id'],float(row['arrival_seconds']),size,kind,kiosk,dine,
            sample('kiosk_dataset',2),sample('food_delay_dataset',3),sample('dining_dataset',4),not bool(row['group_size'])))
    return parties


def interval(values):
    values=np.asarray(values,dtype=float)
    if len(values)<2: return {'n':len(values),'mean':float(values.mean()),'low95':None,'high95':None}
    mean=float(values.mean());half=float(stats.t.ppf(.975,len(values)-1)*stats.sem(values))
    return {'n':len(values),'mean':mean,'low95':mean-half,'high95':mean+half}


def experiment(data_dir,output,config):
    if config['replications']<1 or int(config['replications'])!=config['replications']: raise ValueError('Invalid replications')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    scenarios={'baseline':{},'three_kiosks':{'kiosks':3},'forty_five_seats':{'seats':45},
        'iqr_sensitivity':{'iqr_filter':True},'legacy_kiosk_sample':{'kiosk_dataset':'kiosk_notebook'},
        'unknown_group_size_two':{'missing_group_size':2}}
    runs=[];configs={}
    for name,changes in scenarios.items():
        cfg=config|changes;configs[name]=cfg
        for rep in range(config['replications']):
            seed=config['seed']+rep
            parties=draw_parties(data_dir,cfg,seed)
            result,records,events=run(parties,kiosks=cfg['kiosks'],seats=cfg['seats'],
                horizon=cfg['horizon_seconds'],patience=cfg['seat_patience_seconds'],log_events=rep==0)
            runs.append({'scenario':name,'replication':rep,'seed':seed,**result})
            if rep==0:
                save_csv(output/f'{name}_parties.csv',records)
                save_csv(output/f'{name}_events.csv',events)
    save_csv(output/'replications.csv',runs);save_json(output/'scenarios.json',configs)
    metrics=['customer_restricted_mean_seconds','customer_full_cohort_mean_seconds','customer_food_ready_mean_seconds',
             'customer_kiosk_wait_mean_seconds','customer_seat_wait_mean_seconds','remaining_groups_at_horizon','seat_utilization','kiosk_utilization']
    summary={};paired={};base=[r for r in runs if r['scenario']=='baseline']
    for name in scenarios:
        selected=[r for r in runs if r['scenario']==name]
        summary[name]={m:interval([r[m] for r in selected if r[m] is not None]) for m in metrics}
        if name!='baseline':
            paired[name]={m:interval([r[m]-b[m] for r,b in zip(selected,base) if r[m] is not None and b[m] is not None]) for m in metrics}
    save_json(output/'summary.json',summary);save_json(output/'paired_differences.json',paired)
    save_json(output/'run_metadata.json',{'config':config,'replications_total':len(runs),
        'data_manifest':json.loads((Path(data_dir)/'manifest.json').read_text()),
        'interval_scope':'Student-t Monte Carlo intervals across replications; paired scenario differences use the same seed and per-group uniforms. Input uncertainty excluded.',
        'termination':'Arrivals in [0,H); track occupancy only up to H; drain all accepted parties after H.',
        'tie_order':'Resource completions and food readiness, then seat timeouts, then arrivals; insertion order breaks ties within a priority.',
        'model':'New Python reference; not equivalent to historical ALP.'})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'results/simulation')
    p.add_argument('--config',type=Path,default=ROOT/'config.json');p.add_argument('--replications',type=int)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.replications is not None:cfg['replications']=a.replications
    experiment(a.data,a.output,cfg);print(f'Saved reference simulation to {a.output}')
