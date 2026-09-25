"""Дискретная модель водного баланса и самостоятельная реализация NSGA-II."""
import numpy as np


def generate(seed,days):
    rng=np.random.default_rng(seed)
    return dict(seed=seed,rain=np.round(rng.uniform(0,4,days),3).tolist(),evaporation=np.round(rng.uniform(3,7,days),3).tolist(),initial_moisture=22.,capacity=45.,lower_comfort=20.,upper_comfort=30.)


def simulate(x,data):
    x=np.atleast_2d(x); moisture=np.full(len(x),data['initial_moisture']); history=[]
    deficit=np.zeros(len(x)); excess=np.zeros(len(x))
    for t in range(x.shape[1]):
        raw=moisture+x[:,t]+data['rain'][t]-data['evaporation'][t]
        # Излишек учитывается до отсечения: сток не скрывает перелив.
        deficit+=np.maximum(data['lower_comfort']-raw,0)
        excess+=np.maximum(raw-data['upper_comfort'],0)
        moisture=np.clip(raw,0,data['capacity']); history.append(moisture.copy())
    return np.column_stack([x.sum(1),deficit,excess]),np.array(history).T


def fronts(values):
    """O(n²m): dom[i,j] означает, что i доминирует j."""
    dom=np.all(values[:,None,:]<=values[None,:,:],axis=2)&np.any(values[:,None,:]<values[None,:,:],axis=2)
    count=dom.sum(0); current=np.flatnonzero(count==0); result=[]
    while len(current):
        result.append(current)
        count[current]=-1
        count-=dom[current].sum(0)
        current=np.flatnonzero(count==0)
    return result


def crowding(values,front):
    result=np.zeros(len(front))
    if len(front)<=2: return np.full(len(front),np.inf)
    v=values[front]
    for k in range(v.shape[1]):
        order=np.argsort(v[:,k],kind='stable'); span=np.ptp(v[:,k])
        if span==0: continue
        result[order[[0,-1]]]=np.inf
        result[order[1:-1]]+=(v[order[2:],k]-v[order[:-2],k])/span
    return result


def rank_distance(values):
    ranks=np.empty(len(values),int); distance=np.zeros(len(values))
    for rank,front in enumerate(fronts(values)):
        ranks[front]=rank; distance[front]=crowding(values,front)
    return ranks,distance


def survive(pop,values,n):
    selected=[]
    for front in fronts(values):
        if len(selected)+len(front)<=n: selected.extend(front)
        else:
            order=np.argsort(-crowding(values,front),kind='stable')
            selected.extend(front[order[:n-len(selected)]]); break
    return pop[selected],values[selected]


def scales(data,cfg):
    days=len(data['rain'])
    return np.array([days*cfg['max_irrigation'],days*data['lower_comfort'],days*data['capacity']])


def run(cfg,data,seed,weights=None):
    rng=np.random.default_rng(seed); n,d=cfg['population'],cfg['days']; upper=cfg['max_irrigation']
    pop=rng.uniform(0,upper,(n,d)); values,_=simulate(pop,data); calls=n
    history=[]
    def metric(v): return float(np.min((v/scales(data,cfg)).sum(1)))
    history.append(metric(values))
    for _ in range(cfg['generations']):
        if weights is None:
            ranks,dist=rank_distance(values)
            def parents():
                a,b=rng.integers(n,size=(2,n)); win=(ranks[a]<ranks[b])|((ranks[a]==ranks[b])&(dist[a]>=dist[b])); return pop[np.where(win,a,b)]
        else:
            score=(values/scales(data,cfg))@weights
            def parents():
                ids=rng.integers(n,size=(n,3)); return pop[ids[np.arange(n),score[ids].argmin(1)]]
        a,b=parents(),parents(); alpha=rng.uniform(-.2,1.2,(n,d)); children=np.where(rng.random((n,1))<cfg['crossover_probability'],a+alpha*(b-a),a)
        children+=rng.normal(0,cfg['mutation_sigma'],(n,d))*(rng.random((n,d))<cfg['mutation_probability']); children=np.clip(children,0,upper)
        child_values,_=simulate(children,data); calls+=n
        merged=np.r_[pop,children]; f=np.r_[values,child_values]
        if weights is None: pop,values=survive(merged,f,n)
        else:
            order=np.argsort((f/scales(data,cfg))@weights,kind='stable')[:n]; pop,values=merged[order],f[order]
        history.append(metric(values))
    front=fronts(values)[0]
    # Исключение идентичных генотипов из опубликованного фронта.
    _,ids=np.unique(pop[front],axis=0,return_index=True); front=front[ids]
    return pop[front],values[front],history,calls


def coverage(a,b):
    """Доля b, слабо доминируемая хотя бы одной точкой a."""
    return float(np.any(np.all(a[:,None,:]<=b[None,:,:],axis=2),axis=0).mean())
