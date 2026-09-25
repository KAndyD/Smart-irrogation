"""Сравнение NSGA-II с тремя фиксированными свёртками при равном бюджете."""
import json
import time
import numpy as np
from common import ROOT,setup,write_json,write_csv,summarize,convergence,save_histories,style,plt
from model import generate,run,simulate,scales,coverage


def main():
    cfg,out=setup(); path=ROOT/'data/weather.json'
    data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else generate(cfg['data_seed'],cfg['days'])
    if len(data['rain'])!=cfg['days']: raise ValueError('days не соответствует данным')
    if not path.exists(): write_json(path,data)
    write_json(out/'input_data.json',data)
    rows=[]; histories={}; archive={}; front_rows=[]; comparisons=[]
    methods=[('NSGA_II',None),('weighted_balanced',np.array([1,1,1])/3),('weighted_water',np.array([.8,.1,.1])),('weighted_comfort',np.array([.1,.8,.1]))]
    per_seed={}
    for method,weights in methods:
        histories[method]=[]
        for seed in range(cfg['seed'],cfg['seed']+cfg['runs']):
            start=time.perf_counter(); x,f,h,calls=run(cfg,data,seed,weights)
            rows.append(dict(method=method,seed=seed,front_size=len(f),balanced_score=float(np.min((f/scales(data,cfg)).sum(1))),min_water=float(f[:,0].min()),min_deficit=float(f[:,1].min()),min_excess=float(f[:,2].min()),evaluations=calls,seconds=time.perf_counter()-start)); histories[method].append(h); per_seed[method,seed]=f
            for i,(xx,ff) in enumerate(zip(x,f)): front_rows.append(dict(method=method,seed=seed,solution=i,water=float(ff[0]),deficit=float(ff[1]),excess=float(ff[2]),schedule=json.dumps(xx.tolist())))
            if seed==cfg['seed']: archive[method]=(x,f)
        print(method,'finished',flush=True)
    for seed in range(cfg['seed'],cfg['seed']+cfg['runs']):
        for method,_ in methods[1:]:
            a,b=per_seed['NSGA_II',seed],per_seed[method,seed]
            comparisons.append(dict(method=method,seed=seed,nsga_covers_weighted=coverage(a,b),weighted_covers_nsga=coverage(b,a)))
    write_csv(out/'runs.csv',rows); write_csv(out/'summary.csv',summarize(rows,['front_size','balanced_score','min_water','min_deficit','min_excess','seconds','evaluations'])); write_csv(out/'fronts.csv',front_rows); write_csv(out/'coverage.csv',comparisons)
    save_histories(out/'histories.csv',histories); convergence(out/'convergence.png',histories,ylabel='min Σ нормированных критериев',title='Диагностическая свёртка, не критерий отбора NSGA-II')
    x,f=archive['NSGA_II']; normal=f/scales(data,cfg)
    # Различные представители: сначала вода, затем дефицит, затем баланс.
    chosen=[]
    for order in [np.argsort(f[:,0]),np.argsort(f[:,1]),np.argsort(normal.sum(1))]:
        chosen.append(next(int(i) for i in order if i not in chosen))
    reps={name:dict(seed=cfg['seed'],water=float(f[i,0]),deficit=float(f[i,1]),excess=float(f[i,2]),irrigation=x[i].tolist(),moisture=simulate(x[i],data)[1][0].tolist()) for name,i in zip(['water_saving','comfort','balanced'],chosen)}
    write_json(out/'representatives.json',reps)
    style(); fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    labels=['Вода, мм','Дефицит, мм·сут','Избыток, мм·сут']
    for ax,(i,j) in zip(axs,[(0,1),(0,2),(1,2)]):
        for method,(_,v) in archive.items(): ax.scatter(v[:,i],v[:,j],s=20,label=method,alpha=.7)
        ax.set(xlabel=labels[i],ylabel=labels[j])
    axs[0].legend(fontsize=8); fig.suptitle('Конечные фронты, первый заранее выбранный seed'); fig.tight_layout(); fig.savefig(out/'pareto.png'); plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for name,v in reps.items(): axs[0].plot(range(1,cfg['days']+1),v['irrigation'],label=name); axs[1].plot(range(1,cfg['days']+1),v['moisture'],label=name)
    axs[1].axhspan(data['lower_comfort'],data['upper_comfort'],alpha=.12,color='green'); axs[0].set(ylabel='Полив, мм'); axs[1].set(xlabel='День',ylabel='Запас воды, мм'); axs[0].legend(); fig.tight_layout(); fig.savefig(out/'schedules.png'); plt.close(fig)

if __name__=='__main__': main()
