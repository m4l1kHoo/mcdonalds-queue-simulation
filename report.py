"""Generate a local, self-contained navigation report from computed results."""
from html import escape
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from prepare import ROOT

LABELS={'baseline':'基线：全部有效样本','three_kiosks':'点单机 2 → 3 台','forty_five_seats':'座位 35 → 45 个',
        'iqr_sensitivity':'IQR 筛选敏感性','legacy_kiosk_sample':'改用旧 Notebook 点单时长','unknown_group_size_two':'缺失同行人数按 2 人'}


def table(headers,rows):
    return '<table><thead><tr>'+''.join(f'<th>{escape(str(x))}</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def render(output):
    output=Path(output)
    def load(path):return json.loads((output/path).read_text())
    stats=load('input/summary.json');sim=load('simulation/summary.json');paired=load('simulation/paired_differences.json')
    metadata=load('run.json');cfg=load('simulation/run_metadata.json')['config']
    names=list(paired)
    vals=[paired[n]['customer_full_cohort_mean_seconds'] for n in names]
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    y=np.arange(len(names));means=np.array([v['mean'] for v in vals])
    if all(v['low95'] is not None for v in vals):
        ax.errorbar(means,y,xerr=[means-[v['low95'] for v in vals],[v['high95'] for v in vals]-means],fmt='o',color='#216b87',capsize=4)
    else:ax.scatter(means,y)
    ax.axvline(0,color='#777',linewidth=1);ax.set_yticks(y,names)
    ax.set(xlabel='Paired difference from baseline (seconds per customer group)',title='Full-cohort time in system: conditional Monte Carlo intervals')
    fig.savefig(output/'scenario_comparison.png',dpi=150);plt.close(fig)
    data_rows=[[k,v['total_records'],v['n'],v['invalid_records'],v['unit'],f"{v['mean']:.3f}",v.get('iqr_flagged','不适用')] for k,v in stats.items()]
    sim_rows=[]
    for name,result in sim.items():
        sim_rows.append([LABELS[name],f"{result['customer_full_cohort_mean_seconds']['mean']/60:.3f}",
            f"{result['customer_kiosk_wait_mean_seconds']['mean']:.3f}",f"{result['customer_seat_wait_mean_seconds']['mean']:.3f}",
            f"{result['remaining_groups_at_horizon']['mean']:.2f}"])
    delta_rows=[]
    for name,result in paired.items():
        v=result['customer_full_cohort_mean_seconds']
        interval=f"[{v['low95']:.3f}, {v['high95']:.3f}]" if v['low95'] is not None else '重复次数不足'
        delta_rows.append([LABELS[name],f"{v['mean']:.3f}",interval])
    order=stats['order_completion'];kiosk=stats['kiosk_service_saturday']
    plot_html=''.join(f'<figure><img src="input/{name}.png" alt="{name} 输入诊断"><figcaption>{name}：全部有效观察、筛选比较、有限 Q–Q 分位点及原观察顺序 ACF。</figcaption></figure>'
                      for name in ['order_completion','dining_duration','kiosk_service_saturday','kiosk_notebook','seat_delay_notebook','call2_elapsed'])
    assumptions=''.join(f'<li><strong>{escape(k)}</strong>：{escape(v)}</li>' for k,v in cfg['assumptions'].items())
    html=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>麦当劳排队课程项目重建结果</title><style>
body{{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.75 system-ui,sans-serif;color:#23323d}}h1,h2{{line-height:1.4}}h2{{margin-top:36px}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}}th,td{{border:1px solid #d0dae1;padding:8px;text-align:left;overflow-wrap:anywhere}}th{{background:#edf3f6}}img{{width:100%;height:auto}}figure{{margin:24px 0}}code{{background:#edf3f6;padding:2px 5px}}a{{color:#146985}}.notice{{background:#eff5f8;border-left:4px solid #30788f;padding:14px 18px}}details{{margin:18px 0}}figcaption{{color:#586b78;font-size:14px}}@media(max-width:700px){{table{{display:block;overflow-x:auto}}}}</style></head><body>
<h1>麦当劳排队课程项目重建结果</h1>
<p>课程项目。项目所有者确认主要负责原始数据采集和 AnyLogic 建模。本次数据流程、诊断修复、Python 参考模型、测试与文档属于新增工作。</p>
<div class="notice">Python 数据到结果流程已运行。此报告展示新参考模型的条件性实验，不能视为旧 AnyLogic 结果复现或真实门店改进效果。AnyLogic 材料保留在本地，不在本仓库交付范围内。</div>
<h2>本次实际运行</h2>
<p>使用 Python {escape(metadata['python'])}；固定种子 {cfg['seed']}，6 个情景，每个 {cfg['replications']} 次，共 {cfg['replications']*6} 次运行。到达序列固定为周六表中的 56 组记录，包含顾客、外卖员及一组观察到的离开者。观察窗口 {cfg['horizon_seconds']/60:g} 分钟，随后停止新到达并排空已有顾客。</p>
<p>测试结果：{escape(metadata['tests']['summary'])}。输入校验与代码版本见 <a href="run.json">运行记录</a>，完整假设见 <a href="simulation/scenarios.json">各情景配置</a>。</p>
<h2>数据诊断的实质发现</h2>
<ul><li>原始数值分钟按用户确认乘以 60。表头明确标为秒的列保留秒，时间戳和 Excel 日小数分别解析。</li>
<li>66 条订单到完成记录均保留，平均 {order['mean']:.3f} 秒，最大 {order['maximum']:.0f} 秒。IQR 规则标记 {order['iqr_flagged']} 条；基线没有删除。</li>
<li>8 条周六机器操作记录平均 {kiosk['mean']:.3f} 秒；旧 Notebook 的 45 条记录平均 {stats['kiosk_notebook']['mean']:.3f} 秒，样本来源不能确认相同，分别建模。</li>
<li>叫号系统第二表保留了 3 条负间隔和 10 条缺失间隔。Notebook 对前三条使用了不同正值；新版没有擅自修正或把两个来源混合。</li>
<li>旧 Q–Q 图最后一点取概率 1，产生无穷大。已改为 (i−0.5)/n，图形尾部问题不能直接证明观测异常。</li>
<li>订单到完成序列的一阶样本 ACF 为 {order['acf_by_retained_source_order'][0]:.3f}。新版计算原观察值的 ACF；参考仿真的独立经验抽样没有保留这种相关性，其结果只能在该假设下解释。</li>
<li>订单号前缀 35/39 的比例可以核查，但“手机/机器”类别的含义缺少表头佐证。16/66 只作为未知点单方式的显式暂定参数。</li></ul>
<p>拟合结果包含同一样本内的 AIC/BIC、KS 距离及部分分布的重新拟合自助检验。自助检验按 IID 连续假设计算，未校正观察依赖、舍入和完整记录筛选。<a href="https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.goodness_of_fit.html">统计方法参考</a>。没有以最小指标自动选择唯一“正确分布”。</p>
{table(['数据集','总记录','有效记录','问题记录','单位','有效均值','IQR 标记'],data_rows)}
<p><a href="input/iqr_flags.csv">逐条 IQR 标记</a> · <a href="input/summary.json">统计摘要</a> · <a href="input/fits.json">全部拟合参数与诊断</a></p>
<h2>参考模型实际流程</h2>
<p>观察到达组 → 已记录或按显式概率分配点单方式 → FIFO 点单机 → 经验订单完成延迟 → 取走或 FIFO 等待整组座位 → 就餐 → 离开。外卖员使用独立取餐延迟路径。没有额外厨房容量约束；订单完成延迟已包含历史拥堵，不能再把它当成纯制作时间叠加厨房排队。</p>
<p>等待时间、食物可取时间和系统逗留时间分别记录。系统逗留时间包含就餐，不能称为“排队等待时间”。先保存观察窗口结束时的未完成顾客，再排空计算完整入场群体统计。报告同时保留受窗口限制的停留时间与仅已完成顾客的均值。</p>
{table(['情景','完整入场顾客组平均逗留 / 分钟','使用机器顾客组平均机前等待 / 秒','堂食顾客组平均座位等待 / 秒','窗口末剩余组数'],sim_rows)}
<h2>配对情景比较</h2>
<p>以下差值为“情景减基线”，单位为秒/顾客组。同一重复使用相同随机数。95% 区间仅衡量固定输入及假设下的仿真随机性，不包含采样和模型不确定性。IQR 行改变输入样本，不代表门店运营改善；不同样本来源与缺失人数行也只表示敏感性。</p>
{table(['情景','平均逗留时间差 / 秒','95% Monte Carlo 区间'],delta_rows)}
<figure><img src="scenario_comparison.png" alt="配对情景差异及区间"></figure>
<p><a href="simulation/replications.csv">全部逐次结果</a> · <a href="simulation/baseline_parties.csv">基线第 1 次逐组结果</a> · <a href="simulation/baseline_events.csv">基线第 1 次事件与资源日志</a> · <a href="simulation/paired_differences.json">全部指标配对区间</a></p>
<h2>假设与适用范围</h2><p>原始材料没有足够信息唯一确定预点单剩余时间、放弃等待、初始占座、订单与人数的对应关系。因此参考模型采用配置中列出的假设。跨日期数据被用于不同处理环节，结果是课程模型实验，尚未获得外部观察验证。</p><details><summary>展开全部可配置假设</summary><ul>{assumptions}</ul></details>
<h2>与原 AnyLogic 项目的关系</h2><p>原 ALP 及未运行验证的修订候选保留在本地，不随本仓库发布。原模型中的满座分流、提前占座、条件概率、到达参数及超时解释尚未完成逐项验证；Python 参考模型与原模型的运行等价性也尚未验证。此报告仅记录 Python 实验。</p>
<h2>输入诊断图</h2>{plot_html}
</body></html>'''
    (output/'report.html').write_text(html)
    return output/'report.html'
