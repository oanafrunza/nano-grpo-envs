import json, os
runs=[
  'cosine_zero_period200',
  'prob_zero_p01',
  'hybrid_soft_roundrobin',
  'baseline_default',
  'baseline_fullzero_every10',
  'fullzero_every10_fw13',
  'softmask_every10_wt05',
  'roundrobin_zero_k4',
  'prob_zero_p01',
  'hybrid_soft_roundrobin'
]
summary={}
for r in runs:
    p=os.path.join(r,'run_log.json')
    if not os.path.exists(p):
        print(f'{r}: NO_LOG')
        continue
    try:
        with open(p,'r') as f:
            obj=json.load(f)
    except Exception as e:
        print(f'{r}: PARSE_ERROR {e}')
        continue
    steps=obj.get('steps',{})
    if not steps:
        print(f'{r}: NO_STEPS')
        continue
    last_eval_step=None
    for s in sorted(map(int,steps.keys()), reverse=True):
        se=steps.get(str(s),{})
        if 'eval' in se:
            last_eval_step=s
            break
    if last_eval_step is None:
        print(f'{r}: NO_EVAL')
        continue
    ev=steps[str(last_eval_step)]['eval']
    m=ev.get('metrics',{})
    summary[r]={
        'last_eval_step': last_eval_step,
        'pass_at_1': m.get('pass_at_1'),
        'avg_format_reward': m.get('avg_format_reward'),
        'num_eval_problems': m.get('num_eval_problems'),
    }

# Print summary in a stable order
for r in runs:
    v=summary.get(r)
    if not v:
        continue
    print(f"{r}: step={v['last_eval_step']} pass@1={v['pass_at_1']:.2f} avg_format={v['avg_format_reward']:.3f} n={v['num_eval_problems']}")

# Identify best by pass@1
best=None
for r,v in summary.items():
    if v.get('pass_at_1') is None:
        continue
    if best is None or v['pass_at_1']>best[1]['pass_at_1']:
        best=(r,v)
if best:
    print(f"BEST_BY_PASS@1: {best[0]} pass@1={best[1]['pass_at_1']:.2f}")