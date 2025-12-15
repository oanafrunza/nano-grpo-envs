import json,sys
import io
p='run_log.json'
# stream read last eval metrics by scanning from end
with open(p,'rb') as f:
    f.seek(0,2)
    size=f.tell()
    chunk=1024*1024
    pos=max(0,size-chunk)
    f.seek(pos)
    data=f.read().decode('utf-8','ignore')
# crude parse: find last occurrence of '"eval"' block
last=data.rfind('"eval"')
if last==-1:
    print('NO_EVAL')
    sys.exit(0)
# find opening brace before eval
start=data.rfind('{',0,last)
end=data.find('}', last)
# not robust; instead load full json (may be large)
try:
    obj=json.load(open(p,'r'))
except Exception as e:
    print('PARSE_ERROR',e)
    sys.exit(0)
steps=obj.get('steps',{})
last_step=max([int(s) for s in steps.keys()]) if steps else None
if last_step is None:
    print('NO_STEPS')
    sys.exit(0)
info=steps[str(last_step)]
ev=info.get('eval')
print('LAST_STEP',last_step)
if ev:
    m=ev.get('metrics',{})
    print('PASS', m.get('pass_at_1'))
    print('AVG_FORMAT', m.get('avg_format_reward'))
    print('NUM', m.get('num_eval_problems'))
else:
    tr=info.get('train')
    if tr:
        gens=tr.get('generations',[])
        if gens:
            avg_fmt=sum(g.get('format_reward',0) for g in gens)/len(gens)
            avg_corr=sum(g.get('correct',0) for g in gens)/len(gens)
            print('TRAIN_AVG_FORMAT', avg_fmt)
            print('TRAIN_AVG_CORRECT', avg_corr)
        print('LOSS', tr.get('loss'))