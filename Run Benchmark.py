"""
run_benchmark.py — CSOC-SSC Benchmark Runner
MIT License — Yoon A Limsuwan 2026
github.com/yoonalimsuwan/SSC-SOC-Controlled-Criticality-

Usage:
  python run_benchmark.py --pdb_dir data/casp14/ --labels dataset__3_.csv --out results/
  python run_benchmark.py --pdb_dir data/casp14/ --chain A --max_res 250
"""
import numpy as np, pandas as pd, argparse, os, gzip, warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.stats import pearsonr

THREE2ONE = {
    'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
    'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
    'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
    'TRP':'W','TYR':'Y','MSE':'M','HSD':'H','HSE':'H','SEC':'U',
}

def load_pdb(path, chain, max_res=300):
    coords, seq = [], []
    opener = gzip.open if path.endswith('.gz') else open
    try:
        with opener(path,'rt',errors='ignore') as f:
            seen = set()
            for l in f:
                if not l.startswith('ATOM') or l[12:16].strip()!='CA': continue
                if l[21]!=chain: continue
                key=(int(l[22:26]),l[26])
                if key in seen: continue
                seen.add(key)
                coords.append([float(l[30:38]),float(l[38:46]),float(l[46:54])])
                seq.append(THREE2ONE.get(l[17:20].strip(),'X'))
                if len(coords)>=max_res: break
    except: pass
    return (np.array(coords),''.join(seq)) if coords else (None,'')

def kabsch(P,Q):
    Pc=P-P.mean(0); Qc=Q-Q.mean(0)
    H=Pc.T@Qc; U,S,Vt=np.linalg.svd(H)
    d=np.linalg.det(Vt.T@U.T)
    R=Vt.T@np.diag([1,1,d])@U.T
    Pr=Pc@R.T
    return float(np.sqrt(np.mean(np.sum((Pr-Qc)**2,axis=1)))), Pr

def csoc_kernel(i,j,alpha=2.5,cutoff=12.):
    r=abs(i-j)+1e-4; return (r**(-alpha))*np.exp(-r/cutoff)

def run_v6(coords, alpha=2.5, noise=0.5, seed=1):
    """CSOC-SSC v6: contact map + SSC states + 4-stage energy minimization."""
    n=len(coords); np.random.seed(seed)
    D=cdist(coords,coords); C=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if abs(i-j)>3 and D[i,j]<15.:
                C[i,j]=(1-D[i,j]/15.)*(1+0.3*csoc_kernel(i,j,alpha))
    C/=(C.max(axis=1,keepdims=True)+1e-8)
    # SSC states
    ctr=coords.mean(0); dc=np.linalg.norm(coords-ctr,axis=1)
    s=np.clip(1-(dc-dc.min())/(dc.max()-dc.min()+1e-8),0.05,0.95)
    for t in range(150):
        g=-(C@s)+0.05*(1-2*s); sp=s.copy()
        for i in range(n):
            w=np.array([csoc_kernel(i,j,alpha) for j in range(n)]); w[i]=0
            sp[i]=np.clip(s[i]+0.08*(w@s)/(w.sum()+1e-8),0,1)
        eta=0.04 if t<75 else 0.02
        s=np.clip(s-eta*g+0.1*(sp-s),0,1)
    # F1
    si_sj=np.outer(s,s); tm=C>0; pred=(si_sj>0.5)&tm
    tp=(pred&tm).sum(); fp=(pred&~tm).sum(); fn=(~pred&tm).sum()
    f1=float(2*tp/(2*tp+fp+fn+1e-8))
    # dihedral from coords
    dihs=[]
    for i in range(n-3):
        b1=coords[i+1]-coords[i]; b2=coords[i+2]-coords[i+1]; b3=coords[i+3]-coords[i+2]
        nv1=np.cross(b1,b2); nv2=np.cross(b2,b3)
        n1=np.linalg.norm(nv1)+1e-8; n2=np.linalg.norm(nv2)+1e-8
        cw=np.clip(np.dot(nv1,nv2)/(n1*n2),-1,1)
        m=np.cross(nv1,b2/(np.linalg.norm(b2)+1e-8))
        dihs.append(np.degrees((1. if np.dot(m,nv2)>0 else -1.)*np.arccos(cw)))
    tdih=np.array(dihs)
    disto=[(i,j,D[i,j],0.05) for i in range(n) for j in range(i+4,n) if D[i,j]<15.]
    d_id=float(np.mean([np.linalg.norm(coords[i+1]-coords[i]) for i in range(n-1)]))
    def efn(x,disto_,wb,wd,wc,wa,wdh):
        c=x.reshape(n,3); E=0.; g=np.zeros_like(c)
        for i in range(n-1):
            dv=c[i+1]-c[i]; d=np.linalg.norm(dv)+1e-8; dev=d-d_id
            E+=wb*dev**2; gv=2*wb*dev/d*dv; g[i]-=gv; g[i+1]+=gv
        ci=np.cos(np.radians(111))
        for i in range(n-2):
            b1=c[i+1]-c[i]; b2=c[i+2]-c[i+1]
            n1=np.linalg.norm(b1)+1e-8; n2=np.linalg.norm(b2)+1e-8
            E+=wa*(np.clip(np.dot(b1,b2)/(n1*n2),-1,1)-ci)**2
        for i in range(n-3):
            b1=c[i+1]-c[i]; b2=c[i+2]-c[i+1]; b3=c[i+3]-c[i+2]
            nv1=np.cross(b1,b2); nv2=np.cross(b2,b3)
            n1=np.linalg.norm(nv1)+1e-8; n2=np.linalg.norm(nv2)+1e-8
            E+=wdh*(np.arccos(np.clip(np.dot(nv1,nv2)/(n1*n2),-1,1))-np.radians(abs(tdih[i])))**2
        for (i,j,dt,tol) in disto_:
            dv=c[i]-c[j]; d=np.linalg.norm(dv)+1e-8
            ex=abs(d-dt)-tol
            if ex>0:
                sg=1 if d>dt else -1; E+=wd*ex**2
                gv=2*wd*ex*sg/d*dv; g[i]+=gv; g[j]-=gv
        for i in range(n):
            for j in range(i+3,min(n,i+20)):
                dv=c[i]-c[j]; d=np.linalg.norm(dv)+1e-8
                if d<3.8: dev=3.8-d; E+=wc*dev**2; g[i]+=(-2*wc*dev/d*dv); g[j]-=(-2*wc*dev/d*dv)
        return E,g.ravel()
    c0=coords+np.random.randn(n,3)*noise; r0=kabsch(c0,coords)[0]
    x=minimize(efn,c0.ravel(),args=([],30,0,0,8,0),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-11}).x
    x=minimize(efn,x,args=([],25,0,0,8,5),jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-12}).x
    x=minimize(efn,x,args=(disto,20,5,0,5,5),jac=True,method='L-BFGS-B',options={'maxiter':600,'ftol':1e-13}).x
    x=minimize(efn,x,args=(disto,15,20,80,5,8),jac=True,method='L-BFGS-B',options={'maxiter':800,'ftol':1e-14}).x
    c4=x.reshape(n,3); rmsd,Prot=kabsch(c4,coords)
    Qc=coords-coords.mean(0); per_res=np.sqrt(np.sum((Prot-Qc)**2,axis=1))
    bd=float(np.mean([abs(np.linalg.norm(c4[i+1]-c4[i])-d_id) for i in range(n-1)]))
    nc=sum(1 for i in range(n) for j in range(i+3,n) if np.linalg.norm(c4[i]-c4[j])<3.5)
    Dp=(1-si_sj)*D.max(); mask=np.array([[abs(i-j)>3 and D[i,j]<15. for j in range(n)] for i in range(n)])
    dmae=float(np.mean(np.abs(D[mask]-Dp[mask]))) if mask.sum()>0 else 999.
    return dict(rmsd_init=r0,rmsd_final=rmsd,f1=f1,per_res=per_res,bond_dev=bd,
                n_clash=nc,s=s,dist_mae=dmae,n_below05=int((per_res<0.5).sum()),n_below2=int((per_res<2.).sum()))

def main():
    p=argparse.ArgumentParser(description='CSOC-SSC v6 Benchmark')
    p.add_argument('--pdb_dir',default='data/casp14/')
    p.add_argument('--labels',default='dataset__3_.csv')
    p.add_argument('--out',default='results/')
    p.add_argument('--chain',default='A')
    p.add_argument('--max_res',type=int,default=250)
    p.add_argument('--alpha',type=float,default=2.5)
    p.add_argument('--noise',type=float,default=0.5)
    args=p.parse_args()
    os.makedirs(args.out,exist_ok=True)
    df_all=pd.read_csv(args.labels) if os.path.exists(args.labels) else None
    pdb_files=[f for f in os.listdir(args.pdb_dir) if f.endswith(('.pdb','.pdb.gz','.ent.gz'))]
    print(f"Found {len(pdb_files)} PDB files  chain={args.chain}  max_res={args.max_res}")
    rows=[]
    for fname in sorted(pdb_files):
        pdb_id=fname.split('_')[0].split('.')[0].upper()
        coords,seq=load_pdb(os.path.join(args.pdb_dir,fname),args.chain,args.max_res)
        if coords is None or len(coords)<10: print(f"  {pdb_id}: SKIP"); continue
        n=len(coords)
        print(f"  {pdb_id} n={n}...",end=' ',flush=True)
        r=run_v6(coords,args.alpha,args.noise)
        q3,rsa_r,dih_mae=0.,0.,999.
        if df_all is not None:
            df_p=df_all[(df_all['pdb']==pdb_id)&(df_all['chain']==args.chain)].reset_index(drop=True)
            if len(df_p)>0:
                ss_pred=['H' if r['s'][i]>0.65 else 'C' if r['s'][i]<0.35 else 'B' for i in range(min(n,len(df_p)))]
                ss_true=df_p['q3'].tolist()[:len(ss_pred)]
                q3=sum(a==b for a,b in zip(ss_pred,ss_true))/max(len(ss_true),1)
                rsa_t=df_p['rsa'].values[:n]; rsa_p=1-r['s'][:n]
                if len(rsa_t)>=3:
                    try: rsa_r,_=pearsonr(rsa_p,rsa_t)
                    except: pass
        print(f"RMSD={r['rmsd_final']:.3f}Å F1={r['f1']:.3f} Q3={q3:.2f} <0.5Å={r['n_below05']}/{n}")
        rows.append(dict(pdb=pdb_id,n=n,f1=r['f1'],dist_mae=r['dist_mae'],
                         rmsd_init=r['rmsd_init'],rmsd_final=r['rmsd_final'],
                         q3=q3,rsa_corr=rsa_r,bond_dev=r['bond_dev'],
                         n_clash=r['n_clash'],n_below05=r['n_below05'],n_below2=r['n_below2']))
    df=pd.DataFrame(rows)
    out_csv=os.path.join(args.out,'benchmark_results.csv')
    df.to_csv(out_csv,index=False)
    print(f"\nSaved: {out_csv}")
    print(f"Mean RMSD={df['rmsd_final'].mean():.3f}Å  F1={df['f1'].mean():.3f}  Q3={df['q3'].mean():.3f}")
    print(f"RMSD<0.5Å: {(df['rmsd_final']<0.5).sum()}/{len(df)}")

if __name__=='__main__': main()
