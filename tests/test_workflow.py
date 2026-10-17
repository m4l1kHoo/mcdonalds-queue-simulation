from datetime import datetime, time
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
from scipy import stats

from prepare import decimal_minutes, duration_between, explicit_duration, literal_arrays, ROOT
from analyze import plotting_positions, iqr_mask, samples, valid_values, acf_observations
from simulate import Party, run, draw_parties


def party(id='a',arrival=0,size=1,kiosk=False,dine=False,service=0,food=0,dining=0):
    return Party(id,arrival,size,'customer',kiosk,dine,service,food,dining)


def test_decimal_minutes_not_mmss():
    assert decimal_minutes(1.30)==pytest.approx(78)
    assert decimal_minutes(2.45)==pytest.approx(147)
    assert decimal_minutes(1.95)==pytest.approx(117)
    for x in [-1,float('inf'),float('nan')]:
        with pytest.raises(ValueError):decimal_minutes(x)


def test_durations_do_not_silently_wrap_bad_dates():
    assert duration_between(time(12,59,50),time(13,0,10))==(20,'valid')
    assert duration_between(time(12,0,10),time(12,0,0))==(-10,'negative_interval')
    assert duration_between(datetime(2024,12,7,12),datetime(2024,12,8,12))==(86400,'date_conflict')
    assert explicit_duration('8min8s')==488
    with pytest.raises(ValueError):explicit_duration('1min95s')


def test_notebook_is_never_executed(tmp_path):
    p=tmp_path/'test.ipynb'
    p.write_text(json.dumps({'cells':[{'cell_type':'code','source':['raise RuntimeError("must not execute")\nx=[1,2]\ny=(unknown, other)']}]}))
    assert literal_arrays(p)=={'x':(1,[1,2])}


def test_qq_endpoint_regression():
    for n in [1,15,118]:
        p=plotting_positions(n)
        assert ((p>0)&(p<1)).all()
        assert np.isfinite(stats.gamma.ppf(p,a=2.5,scale=1.018)).all()
    assert np.isinf(stats.gamma.ppf(1,a=2.5))


def test_iqr_flags_do_not_mutate_values():
    x=np.array([1.,2.,3.,4.,100.]);before=x.copy()
    mask,low,high=iqr_mask(x)
    assert mask.tolist()==[True,True,True,True,False]
    np.testing.assert_array_equal(x,before)


def test_acf_observed_order():
    assert acf_observations([1,1,1])==[]
    assert acf_observations([1,2,3,4],1)[0]==pytest.approx(.25)


def test_prepared_counts_and_known_hand_calculation():
    data=samples(ROOT/'data')
    assert len(data['order_completion'])==66
    assert valid_values(data['order_completion'])[0]==125
    assert len(valid_values(data['call2_elapsed']))==49
    assert sum(r['status']=='negative_interval' for r in data['call2_elapsed'])==3
    assert len(data['pick_leave_flag'])==80  # excludes summary row
    assert valid_values(data['dining_duration'])[0]==488
    assert valid_values(data['kiosk_service_saturday'])[0]==pytest.approx(145.77)  # header seconds, not times 60
    assert valid_values(data['kiosk_notebook'])[0]==246  # decimal minutes
    rec=json.loads((ROOT/'data/reconciliation.json').read_text())
    assert rec[0]['exact_matches']==66


def test_known_fcfs_single_kiosk_solution():
    result,rows,events=run([party('a',0,kiosk=True,service=10),party('b',1,kiosk=True,service=10)],kiosks=1,horizon=30,log_events=True)
    assert [r['departure'] for r in rows]==[10,20]
    assert [r['kiosk_wait'] for r in rows]==[0,9]
    assert result['kiosk_utilization']==pytest.approx(20/30)
    assert all(e['busy_kiosks']<=1 for e in events)


def test_parallel_kiosks():
    _,rows,_=run([party('a',0,kiosk=True,service=10),party('b',1,kiosk=True,service=10)],kiosks=2)
    assert [r['departure'] for r in rows]==[10,11]


def test_food_delay_added_once_after_ordering():
    _,rows,_=run([party(kiosk=True,service=10,food=20)])
    assert rows[0]['departure']==30
    assert rows[0]['food_ready_elapsed']==30


def test_strict_fifo_atomic_group_seats():
    ps=[party('a',0,size=2,dine=True,dining=10),party('b',1,size=2,dine=True,dining=10),party('c',2,size=1,dine=True,dining=5)]
    result,rows,_=run(ps,seats=3,horizon=30)
    assert [r['seat_start'] for r in rows]==[0,10,10]
    assert [r['departure'] for r in rows]==[10,20,15]
    assert result['seat_utilization']==pytest.approx(45/90)


def test_timeout_and_completion_tie():
    _,rows,_=run([party('a',0,dine=True,dining=10),party('b',0,dine=True,dining=5)],seats=1,patience=10)
    assert rows[1]['departure_reason']=='dined'
    assert rows[1]['seat_start']==10


def test_timeout_releases_queue_without_seizing():
    result,rows,_=run([party('a',0,dine=True,dining=10),party('b',1,dine=True,dining=5)],seats=1,patience=2)
    assert rows[1]['departure']==3
    assert rows[1]['departure_reason']=='seat_timeout_takeaway'
    assert result['seat_timeout_groups']==1


def test_censoring_and_draining_conservation():
    result,rows,_=run([party('a',0,food=20),party('b',10,food=1)],horizon=10)
    assert result['arrived_groups']==1  # [0,H)
    assert result['remaining_groups_at_horizon']==1
    assert result['departed_groups_by_horizon']==0
    assert result['customer_restricted_mean_seconds']==10
    assert result['customer_full_cohort_mean_seconds']==20
    assert result['customer_completed_only_mean_seconds'] is None


def test_zero_duration_terminates():
    result,rows,_=run([party(kiosk=True,dine=True)],seats=1,kiosks=1)
    assert rows[0]['departure']==0
    assert result['arrived_groups']==result['departed_groups_by_horizon']


def test_invalid_parameters_fail():
    for ps,kwargs in [([party(size=3,dine=True)],{'seats':2}),([party(service=-1)],{}),([party(),party()],{}),([] ,{'kiosks':0})]:
        with pytest.raises(ValueError):run(ps,**kwargs)


def test_trace_randomness_and_end_to_end_invariants():
    cfg=json.loads((ROOT/'config.json').read_text())
    a=draw_parties(ROOT/'data',cfg,123);b=draw_parties(ROOT/'data',cfg,123)
    assert a==b and len(a)==56
    c=draw_parties(ROOT/'data',cfg|{'seats':45},123)
    assert c==a  # resource scenario does not alter input draws
    for seed in range(5):
        result,rows,events=run(draw_parties(ROOT/'data',cfg,seed),log_events=True)
        assert result['arrived_groups']==result['departed_groups_by_horizon']+result['remaining_groups_at_horizon']
        assert result['arrived_people']==result['departed_people_by_horizon']+result['remaining_people_at_horizon']
        assert 0<=result['seat_utilization']<=1 and 0<=result['kiosk_utilization']<=1
        assert all(r['time_in_system']>=0 for r in rows)
        assert all(e['arrived_groups']>=e['departed_groups'] for e in events)
