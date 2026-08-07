"""
plot_figure5_dependence_v2.py — Figure 5 v2: 9 panels (3x3) — FIXED
=====================================================================
(a)-(e): SHAP dependence  (f): Alignment  (g): Shift  (h): Rank  (i): Sign
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))
SHAP_CSV = os.path.join(BASE, "Results", "Knowledge_Assimilation", "SHAP_Relationships", "signed_shap_samples.csv")
SAVE = os.path.join(BASE, "Graphs", "Figure5_Dependence")
os.makedirs(SAVE, exist_ok=True)

COLOR_BEFORE = "#5DA5DA"; COLOR_AFTER = "#C91511"; COLOR_GREEN = "#6EAA5E"
DARK = "#333333"; GRAY = "#8C92AC"
FS = 14
plt.rcParams.update({"font.family":"Arial","font.size":FS,"axes.unicode_minus":False,
    "axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
    "xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

TARGETS = ["LOI","UL94_Rating","THR","TSP","Flexural_Strength"]
DISP = {"LOI":"LOI","UL94_Rating":"UL-94","THR":"THR","TSP":"TSP","Flexural_Strength":"Flexural"}
FEATURES = {"LOI":"B01[C-P]_FR","UL94_Rating":"F01[C-P]_FR","THR":"Eta_sh_x_Curing",
            "TSP":"Mor32e_FR","Flexural_Strength":"MDEN-22_FR"}

def load_data():
    return pd.read_csv(SHAP_CSV)

def draw_dependence(ax, df, target):
    feat = FEATURES[target]
    sub = df[(df["target"]==target)&(df["feature"]==feat)]
    before = sub[sub["track"]=="before"]; after = sub[sub["track"]=="after"]
    xb = before["feature_value_raw"].values; yb = before["shap_value"].values
    xa = after["feature_value_raw"].values; ya = after["shap_value"].values
    x_all = np.concatenate([xb,xa]) if len(xb)>0 and len(xa)>0 else (xb if len(xb)>0 else xa)
    x_range = np.ptp(x_all) if len(x_all)>1 else 1.0
    offset = max(x_range * 0.015, 0.005)
    unique_vals = len(set(np.round(x_all,6)))
    is_binary = (unique_vals <= 2)
    zero_frac = np.mean(x_all==0) if len(x_all)>0 else 0

    if is_binary:
        for i,(xvals,yvals,color,off) in enumerate([(xb,yb,COLOR_BEFORE,-offset),(xa,ya,COLOR_AFTER,+offset)]):
            jitter = np.random.default_rng(i).uniform(-offset*0.3,offset*0.3,size=len(xvals))
            ax.scatter(xvals+jitter+off,yvals,s=10,c=color,alpha=0.3,edgecolors="none")
            for val in sorted(set(xvals)):
                mask=xvals==val; n=mask.sum()
                if n>=3:
                    med=np.median(yvals[mask])
                    bs=[np.median(yvals[mask][np.random.default_rng(j).integers(0,n,n)]) for j in range(500)]
                    ci_lo,ci_hi=np.percentile(bs,[2.5,97.5]); w=offset*0.8
                    ax.plot([val+off-w,val+off+w],[med,med],color=color,linewidth=2.5)
                    ax.plot([val+off,val+off],[ci_lo,ci_hi],color=color,linewidth=1.2,alpha=0.6)
    elif zero_frac>0.3:
        for i,(xvals,yvals,color) in enumerate([(xb,yb,COLOR_BEFORE),(xa,ya,COLOR_AFTER)]):
            off=-offset if i==0 else +offset
            jitter=np.random.default_rng(i).uniform(-offset*0.3,offset*0.3,size=len(xvals))
            ax.scatter(xvals+jitter+off,yvals,s=6,c=color,alpha=0.2,edgecolors="none")
            mask0=xvals==0; n0=mask0.sum()
            if n0>=3:
                med=np.median(yvals[mask0])
                bs=[np.median(yvals[mask0][np.random.default_rng(j).integers(0,n0,n0)]) for j in range(500)]
                ci_lo,ci_hi=np.percentile(bs,[2.5,97.5]); w=offset*0.8
                ax.plot([off-w,off+w],[med,med],color=color,linewidth=2.5)
                ax.plot([off,off],[ci_lo,ci_hi],color=color,linewidth=1.2,alpha=0.6)
            nz=xvals[xvals>0]; y_nz=yvals[xvals>0]
            if len(nz)>=10:
                deciles=np.percentile(nz,np.arange(0,101,20)); bx=[];bm=[];blo=[];bhi=[]
                for bi in range(len(deciles)-1):
                    mask=(nz>=deciles[bi])&(nz<deciles[bi+1]); n=mask.sum()
                    if n>=3:
                        bx.append((deciles[bi]+deciles[bi+1])/2); m=np.median(y_nz[mask])
                        bs=[np.median(y_nz[mask][np.random.default_rng(j).integers(0,n,n)]) for j in range(500)]
                        bm.append(m); blo.append(np.percentile(bs,2.5)); bhi.append(np.percentile(bs,97.5))
                if len(bx)>1: ax.plot(bx,bm,color=color,linewidth=2.0); ax.fill_between(bx,blo,bhi,color=color,alpha=0.1)
    elif unique_vals<=30:
        for i,(xvals,yvals,color,off) in enumerate([(xb,yb,COLOR_BEFORE,-offset),(xa,ya,COLOR_AFTER,+offset)]):
            jitter=np.random.default_rng(i).uniform(-offset*0.3,offset*0.3,size=len(xvals))
            ax.scatter(xvals+jitter+off,yvals,s=6,c=color,alpha=0.25,edgecolors="none")
            for val in sorted(set(xvals)):
                mask=xvals==val; n=mask.sum()
                if n>=3:
                    med=np.median(yvals[mask])
                    bs=[np.median(yvals[mask][np.random.default_rng(j).integers(0,n,n)]) for j in range(500)]
                    ci_lo,ci_hi=np.percentile(bs,[2.5,97.5]); w=offset*0.8
                    ax.plot([val+off-w,val+off+w],[med,med],color=color,linewidth=2.5)
                    ax.plot([val+off,val+off],[ci_lo,ci_hi],color=color,linewidth=1.2,alpha=0.6)
    else:
        for i,(xvals,yvals,color) in enumerate([(xb,yb,COLOR_BEFORE),(xa,ya,COLOR_AFTER)]):
            off=-offset if i==0 else +offset
            jitter=np.random.default_rng(i).uniform(-offset*0.3,offset*0.3,size=len(xvals))
            ax.scatter(xvals+jitter+off,yvals,s=5,c=color,alpha=0.2,edgecolors="none")
            deciles=np.percentile(xvals,np.arange(0,101,10)); bx=[];bm=[];blo=[];bhi=[]
            for bi in range(len(deciles)-1):
                mask=(xvals>=deciles[bi])&(xvals<deciles[bi+1]); n=mask.sum()
                if n>=3:
                    bx.append((deciles[bi]+deciles[bi+1])/2); m=np.median(yvals[mask])
                    bs=[np.median(yvals[mask][np.random.default_rng(j).integers(0,n,n)]) for j in range(500)]
                    bm.append(m); blo.append(np.percentile(bs,2.5)); bhi.append(np.percentile(bs,97.5))
            if len(bx)>1: ax.plot(bx,bm,color=color,linewidth=2.0); ax.fill_between(bx,blo,bhi,color=color,alpha=0.1)
    ax.axhline(0,color=DARK,linewidth=0.7,linestyle="--")
    ax.set_xlabel(feat,fontweight="bold",fontsize=FS-2)
    ylabel = "SHAP contribution to V-0 log-odds" if target=="UL94_Rating" else "SHAP value"
    ax.set_ylabel(ylabel,fontweight="bold",fontsize=FS-1)

def panel_f_alignment(ax, df):
    results = []
    for target in TARGETS:
        feat=FEATURES[target]; sub=df[(df["target"]==target)&(df["feature"]==feat)&(df["track"]=="after")]
        if len(sub)==0: continue
        med_shap=sub["shap_value"].median()
        yh=sub.loc[sub["shap_value"]>=med_shap,"y_true"].values
        yl=sub.loc[sub["shap_value"]<med_shap,"y_true"].values
        if len(yh)==0 or len(yl)==0: continue
        if target=="UL94_Rating": prob=0.5*np.mean((yh[:,None]>yl[None,:]).flatten())+0.5*np.mean((yh[:,None]==yl[None,:]).flatten())*0.5
        elif target in["THR","TSP"]: prob=np.mean((yl[:,None]<yh[None,:]).flatten())
        else: prob=np.mean((yh[:,None]>yl[None,:]).flatten())
        bs=[]; rng=np.random.default_rng(42); n=len(sub)
        for _ in range(1000):
            idx=rng.integers(0,n,n); hs=sub.iloc[idx]; hm=hs["shap_value"].median()
            h_vals=hs[hs["shap_value"]>=hm]["y_true"].values; l_vals=hs[hs["shap_value"]<hm]["y_true"].values
            if len(h_vals)>0 and len(l_vals)>0:
                if target=="UL94_Rating": b=np.mean((h_vals[:,None]>l_vals[None,:]).flatten())+0.5*np.mean((h_vals[:,None]==l_vals[None,:]).flatten())
                elif target in["THR","TSP"]: b=np.mean((l_vals[:,None]<h_vals[None,:]).flatten())
                else: b=np.mean((h_vals[:,None]>l_vals[None,:]).flatten()); bs.append(b)
        ci_lo,ci_hi=np.percentile(bs,[2.5,97.5]) if len(bs)>0 else(0.5,0.5)
        results.append({"target":target,"prob":prob,"ci_lo":ci_lo,"ci_hi":ci_hi})
    res=pd.DataFrame(results); y=np.arange(len(res))[::-1]
    for i,(_,row) in enumerate(res.iterrows()):
        c=DARK
        ax.plot([row["ci_lo"],row["ci_hi"]],[y[i],y[i]],color=c,linewidth=2.5,alpha=0.7)
        ax.scatter([row["prob"]],[y[i]],s=80,c=c,zorder=5,edgecolors="white",linewidth=1)
    ax.axvline(0.5,color=DARK,linewidth=0.7,linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels([DISP[t] for t in res["target"]],fontsize=FS-1,fontweight="bold")
    ax.set_xlabel("P(SHAP regime = better perf.)",fontweight="bold",fontsize=FS-1)
    ax.set_xlim(0.45,0.85); ax.grid(axis="x",linestyle=":",alpha=0.3)
    ax.set_title("Empirical alignment",fontweight="bold",fontsize=FS,color=DARK)

def panel_g_shift(ax, df):
    results=[]
    for target in TARGETS:
        feat=FEATURES[target]; sub=df[(df["target"]==target)&(df["feature"]==feat)]
        b=sub[sub["track"]=="before"]["shap_value"].values; a=sub[sub["track"]=="after"]["shap_value"].values
        if len(b)==0 or len(a)==0: continue
        num=np.mean(np.abs(a-b)); den=np.mean((np.abs(a)+np.abs(b))/2)
        results.append({"target":target,"shift":num/den*100})
    res=pd.DataFrame(results); x=np.arange(len(res))
    colors=[COLOR_AFTER if v>15 else COLOR_BEFORE for v in res["shift"]]
    ax.barh(x,res["shift"].values,color=colors,edgecolor="white",linewidth=0.5)
    for i,(_,row) in enumerate(res.iterrows()):
        ax.text(row["shift"]+0.5,i,f"{row['shift']:.1f}%",va="center",fontsize=FS-1,fontweight="bold",color=DARK)
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels([DISP[t] for t in res["target"]],fontsize=FS-1,fontweight="bold")
    ax.set_xlabel("Relative SHAP shift (%)",fontweight="bold",fontsize=FS-1)
    ax.grid(axis="x",linestyle=":",alpha=0.3)
    ax.set_title("Relative contribution shift",fontweight="bold",fontsize=FS,color=DARK)

def panel_h_rank(ax, df):
    results=[]
    for target in TARGETS:
        feat=FEATURES[target]; sub=df[(df["target"]==target)&(df["feature"]==feat)]
        b=sub[sub["track"]=="before"][["sample_id","shap_value"]].set_index("sample_id")
        a=sub[sub["track"]=="after"][["sample_id","shap_value"]].set_index("sample_id")
        common=b.index.intersection(a.index)
        if len(common)<3: continue
        rho,_=spearmanr(b.loc[common,"shap_value"],a.loc[common,"shap_value"])
        results.append({"target":target,"rho":rho})
    res=pd.DataFrame(results); y=np.arange(len(res))[::-1]
    for i,(_,row) in enumerate(res.iterrows()):
        c=COLOR_BEFORE; ax.scatter([row["rho"]],[y[i]],s=80,c=c,zorder=5,edgecolors="white",linewidth=1)
        ax.text(row["rho"]+0.001,y[i],f"{row['rho']:.3f}",va="center",fontsize=FS-1,fontweight="bold",color=DARK)
    ax.set_yticks(y); ax.set_yticklabels([DISP[t] for t in res["target"]],fontsize=FS-1,fontweight="bold")
    ax.set_xlabel("Spearman rho",fontweight="bold",fontsize=FS-1); ax.set_xlim(0.95,1.001)
    ax.grid(axis="x",linestyle=":",alpha=0.3)
    ax.set_title("Before-after rank stability",fontweight="bold",fontsize=FS,color=DARK)

def panel_i_sign(ax, df):
    results=[]
    for target in TARGETS:
        feat=FEATURES[target]; sub=df[(df["target"]==target)&(df["feature"]==feat)]
        b=sub[sub["track"]=="before"][["sample_id","shap_value"]].set_index("sample_id")
        a=sub[sub["track"]=="after"][["sample_id","shap_value"]].set_index("sample_id")
        common=b.index.intersection(a.index)
        if len(common)==0: continue
        bv=b.loc[common,"shap_value"].values; av=a.loc[common,"shap_value"].values
        same=np.sum(np.sign(bv)==np.sign(av)); p2n=np.sum((bv>0)&(av<0)); n2p=np.sum((bv<0)&(av>0))
        results.append({"target":target,"same":same,"p2n":p2n,"n2p":n2p,"n":len(common)})
    res=pd.DataFrame(results); y=np.arange(len(res))[::-1]; cum=np.zeros(len(res))
    for state,color,label in[("same",GRAY,"Same sign"),("p2n",COLOR_AFTER,"Pos->Neg"),("n2p",COLOR_BEFORE,"Neg->Pos")]:
        vals=np.array([res[state].iloc[i]/res["n"].iloc[i]*100 for i in range(len(res))])
        ax.barh(y,vals,left=cum,color=color,edgecolor="white",linewidth=0.5,label=label)
        for i in range(len(res)):
            if vals[i]>0.5:
                ax.text(cum[i]+vals[i]/2,y[i],f"{vals[i]:.1f}%",ha="center",va="center",
                        fontsize=FS-1,fontweight="bold",color="white" if vals[i]>25 else DARK)
        cum+=vals
    for i in range(len(res)):
        ax.text(102,y[i],f"n={res['n'].iloc[i]}",va="center",fontsize=FS-3,color=DARK)
    ax.set_yticks(y); ax.set_yticklabels([DISP[t] for t in res["target"]],fontsize=FS-1,fontweight="bold")
    ax.set_xlabel("Fraction of samples (%)",fontweight="bold",fontsize=FS-1); ax.set_xlim(0,115)
    ax.legend(frameon=True,fancybox=False,edgecolor="#777",fontsize=FS-3,loc="upper center",ncol=3)
    ax.set_title("Sign transitions",fontweight="bold",fontsize=FS,color=DARK)

def main():
    df=load_data()
    fig=plt.figure(figsize=(28,28),dpi=300)
    gs=fig.add_gridspec(3,3,hspace=0.35,wspace=0.30)
    axes=[fig.add_subplot(gs[r,c]) for r in range(3) for c in range(3)]
    for ax in axes: ax.set_box_aspect(1)
    for idx,target in enumerate(TARGETS):
        draw_dependence(axes[idx],df,target)
        axes[idx].set_title(DISP[target],fontweight="bold",fontsize=FS+2)
        axes[idx].text(-0.10,1.02,f"({chr(97+idx)})",transform=axes[idx].transAxes,fontsize=FS+4,fontweight="bold")
        if idx==0: axes[idx].legend(handles=[Patch(facecolor=COLOR_BEFORE,edgecolor="white",label="Before"),
                                              Patch(facecolor=COLOR_AFTER,edgecolor="white",label="After")],
                                     frameon=True,fancybox=False,edgecolor="#777",fontsize=FS-2,loc="upper right")
    panel_f_alignment(axes[5],df); axes[5].text(-0.10,1.02,"(f)",transform=axes[5].transAxes,fontsize=FS+4,fontweight="bold")
    panel_g_shift(axes[6],df); axes[6].text(-0.10,1.02,"(g)",transform=axes[6].transAxes,fontsize=FS+4,fontweight="bold")
    panel_h_rank(axes[7],df); axes[7].text(-0.10,1.02,"(h)",transform=axes[7].transAxes,fontsize=FS+4,fontweight="bold")
    panel_i_sign(axes[8],df); axes[8].text(-0.10,1.02,"(i)",transform=axes[8].transAxes,fontsize=FS+4,fontweight="bold")
    for fmt in["png","pdf","svg"]:
        fig.savefig(os.path.join(SAVE,f"Figure5_v2.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print(f"[OK] v2 -> {SAVE}")

if __name__=="__main__": main()
