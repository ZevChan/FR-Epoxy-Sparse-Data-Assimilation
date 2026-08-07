"""
plot_figure4_5_combined.py — Figure 4+5 merged: 3x3 layout (print-ready)
=========================================================================
Target ~180x200mm at 300dpi. FS=22 for ~8pt at final size.
"""
import os, warnings
BASE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

KA = os.path.join(BASE, "Results", "Knowledge_Assimilation")
R1 = os.path.join(BASE, "Results")
SAVE = os.path.join(BASE, "Graphs", "Figure4_5_Combined")
os.makedirs(SAVE, exist_ok=True)

# ── Colors (red-blue scheme) ──
COLOR_BEFORE = "#5DA5DA"   # Near / XGBoost / Inert / Jaccard
COLOR_AFTER  = "#C91511"   # Far / ExtraTrees / Destructive / Spearman
COLOR_MID    = "#8C92AC"   # Middle / neutral
COLOR_GREEN  = "#6EAA5E"   # Stable
COLOR_PURPLE = "#9370DB"   # Plastic
GRAY = "#8C92AC"; DARK = "#333333"
FS = 22  # base font, final ~8pt after scaling

plt.rcParams.update({"font.family":"Arial","font.size":FS,"axes.unicode_minus":False,
    "axes.linewidth":0.8,"xtick.major.width":0.8,"ytick.major.width":0.8,
    "xtick.direction":"in","ytick.direction":"in","xtick.major.size":5,"ytick.major.size":5,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

TARGETS = ["LOI","UL94_Rating","THR","TSP","Flexural_Strength"]
DISP = {"LOI":"LOI","UL94_Rating":"UL-94","THR":"THR","TSP":"TSP","Flexural_Strength":"Flexural"}
TOLERANCE = {"LOI":0.01,"UL94_Rating":0.02,"THR":0.01,"TSP":0.01,"Flexural_Strength":0.01}

def load():
    d = {}
    d["regions"] = pd.read_csv(os.path.join(KA,"Propagation","propagation_regions.csv"))
    d["samples"] = pd.read_csv(os.path.join(KA,"Propagation","propagation_samples.csv"))
    for k,f in [("phase","Plasticity/utility_plasticity_phase_map.csv"),
                ("sens","ModelSensitivity/model_family_sensitivity.csv")]:
        fp = os.path.join(KA,f)
        if os.path.exists(fp): d[k] = pd.read_csv(fp)
    for k,f in [("sim","SHAP_Stability/shap_ranking_similarity.csv"),
                ("fm","Frozen/frozen_metrics.csv")]:
        fp = os.path.join(R1,f)
        if os.path.exists(fp): d[k] = pd.read_csv(fp)
    return d

def compute_update_states(data):
    """Classify each seed per target from frozen_metrics + shap_ranking_similarity."""
    fm = data["fm"]; sim = data["sim"]
    frozen = fm[fm["protocol"]=="frozen"]
    frozen_sim = sim[sim["protocol"]=="frozen"]
    states = {}
    for t in TARGETS:
        counts = {"Inert":0,"Stable":0,"Destructive":0,"Plastic":0}
        for seed in sorted(frozen["seed"].unique()):
            frow = frozen[(frozen["target"]==t)&(frozen["seed"]==seed)]
            srow = frozen_sim[(frozen_sim["target"]==t)&(frozen_sim["seed"]==seed)]
            if len(frow)==0 or len(srow)==0: continue
            util = frow["improvement_delta"].values[0]
            tol = TOLERANCE.get(t,0.01)
            nu = util/tol
            plast = 1.0 - srow["top10_jaccard"].values[0]
            if nu < -1: state = "Destructive"
            elif nu > 1:
                state = "Plastic" if plast >= 0.5 else "Stable"
            else: state = "Inert"
            counts[state] += 1
        states[t] = [counts["Inert"],counts["Stable"],counts["Destructive"],counts["Plastic"]]
    return states

# ═══════════ PANELS ═══════════

def p_a_zone(ax, data):
    regions = data["regions"]; r42 = regions[regions["seed"]==42]
    samples = data["samples"]; s42 = samples[samples["seed"]==42]
    for i, t in enumerate(TARGETS):
        dists = s42[s42["target"]==t]["distance_to_nearest_experiment"].dropna().values
        if len(dists)==0: continue
        xs = np.linspace(dists.min(),dists.max(),200)
        try:
            kde = gaussian_kde(dists); ys = kde(xs); ys = ys/ys.max()*0.4
            ax.plot(xs,ys+i,color=DARK,linewidth=1.2,alpha=0.7)
        except: pass
        sub_r = r42[r42["target"]==t]
        if len(sub_r)>0:
            dn = sub_r[sub_r["neighborhood"]=="near"]["distance_max"].values
            dm = sub_r[sub_r["neighborhood"]=="middle"]["distance_max"].values
            if len(dn)>0: ax.axvline(dn[0],ymin=(i-0.3)/5,ymax=(i+0.3)/5,color=COLOR_BEFORE,linewidth=1.5,linestyle="--",alpha=0.6)
            if len(dm)>0: ax.axvline(dm[0],ymin=(i-0.3)/5,ymax=(i+0.3)/5,color=COLOR_AFTER,linewidth=1.5,linestyle="--",alpha=0.6)
    ax.set_yticks(range(5)); ax.set_yticklabels([DISP[t] for t in TARGETS],fontsize=FS-2,fontweight="bold"); ax.invert_yaxis()
    ax.set_xlabel("Distance to nearest experiment",fontweight="bold",fontsize=FS-2)
    ax.legend(handles=[Line2D([0],[0],color=COLOR_BEFORE,linestyle="--",lw=1.5,label="Near/Mid"),
                       Line2D([0],[0],color=COLOR_AFTER,linestyle="--",lw=1.5,label="Mid/Far")],
              frameon=True,fancybox=False,edgecolor="#777",fontsize=FS-6,loc="lower right")
    ax.text(-0.08,1.02,"(a)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def p_b_heatmap(ax, data):
    regions = data["regions"]; r42 = regions[regions["seed"]==42]
    zones = ["near","middle","far"]; mat = np.zeros((5,3)); annot = []
    for i, t in enumerate(TARGETS):
        ra = []
        for j, z in enumerate(zones):
            sub = r42[(r42["target"]==t)&(r42["neighborhood"]==z)]
            if len(sub)>0:
                bl=sub["mean_loss_before"].values[0]; al=sub["mean_loss_after"].values[0]
                rel=(bl-al)/bl if abs(bl)>1e-8 else 0; mat[i,j]=rel; ra.append(f"{rel:+.3f}")
            else: ra.append("")
        annot.append(ra)
    vmax = max(abs(np.min(mat)),abs(np.max(mat))) if np.max(np.abs(mat))>0 else 0.01
    cmap = LinearSegmentedColormap.from_list("rg",[COLOR_AFTER,"#E8E8E8",COLOR_BEFORE])
    im = ax.imshow(mat,cmap=cmap,vmin=-vmax,vmax=vmax,aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Near","Middle","Far"],fontsize=FS-2,fontweight="bold")
    ax.set_yticks(range(5)); ax.set_yticklabels([DISP[t] for t in TARGETS],fontsize=FS-2,fontweight="bold")
    for i in range(5):
        for j in range(3):
            ax.text(j,i,annot[i][j],ha="center",va="center",fontsize=FS-2,fontweight="bold",
                    color="white" if abs(mat[i,j])>vmax*0.5 else "black")
    # Inline colorbar at bottom
    cbar_ax = ax.inset_axes([0.15,-0.15,0.7,0.04])
    cbar = plt.colorbar(im,cax=cbar_ax,orientation="horizontal")
    cbar.set_label("Rel. loss improvement",fontsize=FS-5,labelpad=2); cbar.ax.tick_params(labelsize=FS-6)
    ax.text(-0.08,1.02,"(b)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def p_c_fraction(ax, data):
    """Fraction improved per zone — full y-axis, no broken axis."""
    regions = data["regions"]; r42 = regions[regions["seed"]==42]
    zones = ["near","middle","far"]; x = np.arange(5); w = 0.22
    for j, z in enumerate(zones):
        vals = []
        for t in TARGETS:
            sub = r42[(r42["target"]==t)&(r42["neighborhood"]==z)]
            vals.append(sub["fraction_improved"].values[0] if len(sub)>0 else 0)
        ax.bar(x+w*(j-1),vals,w,color=[COLOR_BEFORE,COLOR_MID,COLOR_AFTER][j],edgecolor="white",linewidth=0.4)
    ax.axhline(0.5,color=DARK,linewidth=0.7,linestyle="--",alpha=0.5)
    ax.annotate("50% improved",xy=(4.5,0.51),fontsize=FS-5,color=DARK,fontstyle="italic",ha="right")
    ax.set_xticks(x); ax.set_xticklabels([DISP[t] for t in TARGETS],rotation=20,fontsize=FS-2,fontweight="bold")
    ax.set_ylabel("Fraction improved",fontweight="bold",fontsize=FS-2); ax.set_ylim(0,0.70)
    ax.legend(handles=[Patch(facecolor=COLOR_BEFORE,edgecolor="white",label="Near"),
                       Patch(facecolor=COLOR_MID,edgecolor="white",label="Middle"),
                       Patch(facecolor=COLOR_AFTER,edgecolor="white",label="Far")],
              frameon=True,fancybox=False,edgecolor="#777",fontsize=FS-6,loc="lower left")
    ax.grid(axis="y",linestyle=":",alpha=0.3)
    ax.text(-0.08,1.02,"(c)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def p_d_heatmap(ax, data):
    """Normalized median error improvement: 4 targets x 5 distance quintile bins."""
    samples = data["samples"]; s42 = samples[samples["seed"]==42]
    reg_t = ["LOI","THR","TSP","Flexural_Strength"]
    # Compute target-level before MAE for normalization
    target_mae = {}
    for t in reg_t:
        sub = s42[s42["target"]==t]
        target_mae[t] = sub["abs_error_before"].mean()

    mat = np.zeros((4,5)); annot = []
    for i, t in enumerate(reg_t):
        sub = s42[s42["target"]==t]
        dist = sub["distance_percentile"].dropna().values
        delta = (sub["abs_error_before"] - sub["abs_error_after"]).dropna().values
        bins = np.percentile(dist, [0,20,40,60,80,100])
        row_annot = []
        for bi in range(5):
            mask = (dist >= bins[bi]) & (dist < bins[bi+1])
            if mask.sum() > 0:
                med_imp = np.median(delta[mask])
                norm = med_imp / target_mae[t] * 100
            else:
                norm = 0
            mat[i, bi] = norm
            row_annot.append(f"{norm:+.1f}%")
        annot.append(row_annot)
    vmax = max(abs(np.min(mat)), abs(np.max(mat)), 2.5)
    cmap = LinearSegmentedColormap.from_list("rg", [COLOR_AFTER, "#E8E8E8", COLOR_BEFORE])
    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"], fontsize=FS-3, fontweight="bold", rotation=15)
    ax.set_yticks(range(4))
    ax.set_yticklabels([DISP[t] for t in reg_t], fontsize=FS-2, fontweight="bold")
    for i in range(4):
        for j in range(5):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=FS-2, fontweight="bold",
                    color="white" if abs(mat[i,j]) > vmax*0.5 else "black")
    cbar_ax = ax.inset_axes([0.15, -0.15, 0.7, 0.04])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Norm. median imp. (%)", fontsize=FS-5, labelpad=2)
    cbar.ax.tick_params(labelsize=FS-6)
    ax.text(-0.08, 1.02, "(d)", transform=ax.transAxes, fontsize=FS+4, fontweight="bold")

def p_e_phase(ax, data):
    """Seed-42 utility-plasticity map with labels for both model families."""

    # Load and filter
    phase = data["phase"].copy()
    sens = data["sens"].copy()

    p42 = phase[(phase["seed"] == 42) & (phase["target"].isin(TARGETS))].copy()
    s42 = sens[(sens["seed"] == 42) & (sens["target"].isin(TARGETS))].copy()

    # Explicitly retain ExtraTrees only
    if "model_family" in s42.columns:
        s42 = s42[s42["model_family"] == "ExtraTrees"].copy()

    p42 = p42.drop_duplicates(subset="target", keep="first")
    s42 = s42.drop_duplicates(subset="target", keep="first")

    def get_point(df, target):
        row = df[df["target"] == target]
        if row.empty:
            return None
        row = row.iloc[0]
        return float(row["utility"]) / TOLERANCE.get(target, 0.01), float(row["plasticity"])

    xb_points = {t: get_point(p42, t) for t in TARGETS}
    et_points = {t: get_point(s42, t) for t in TARGETS}

    all_points = [p for p in list(xb_points.values()) + list(et_points.values()) if p is not None]
    all_x = [p[0] for p in all_points]
    all_y = [p[1] for p in all_points]

    xmin = min(-1.5, min(all_x) - 0.30)
    xmax = max(3.0, max(all_x) + 0.30)
    ymin = -0.05
    ymax = max(0.72, max(all_y) + 0.18)

    # Operational regions
    ax.axvspan(xmin, -1, facecolor=COLOR_AFTER, alpha=0.045, zorder=0)
    ax.axvspan(-1, 1, facecolor=GRAY, alpha=0.035, zorder=0)
    ax.fill_between([1, xmax], [ymin, ymin], [0.5, 0.5], facecolor=COLOR_GREEN, alpha=0.055, zorder=0)
    ax.fill_between([1, xmax], [0.5, 0.5], [ymax, ymax], facecolor=COLOR_PURPLE, alpha=0.045, zorder=0)

    ax.axvline(-1, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.75, zorder=1)
    ax.axvline(1, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.75, zorder=1)
    ax.axhline(0.5, color=GRAY, linewidth=0.7, linestyle=":", alpha=0.65, zorder=1)

    # Connect corresponding model-family points
    for t in TARGETS:
        xb = xb_points[t]; et = et_points[t]
        if xb is not None and et is not None:
            ax.plot([xb[0], et[0]], [xb[1], et[1]], color="#999999", linewidth=0.7, alpha=0.55, zorder=2)

    # Scatter points
    for t in TARGETS:
        xb = xb_points[t]
        if xb is not None:
            ax.scatter(xb[0], xb[1], s=145, c=COLOR_BEFORE, marker="o", edgecolors="white", linewidth=1.0, zorder=5)
        et = et_points[t]
        if et is not None:
            ax.scatter(et[0], et[1], s=145, c=COLOR_AFTER, marker="s", edgecolors="white", linewidth=1.0, zorder=5)

    # Model-specific label offsets, in display points
    offsets = {
        "LOI": {"XGB": (-30, 14), "ET": (30, 14)},
        "UL94_Rating": {"XGB": (-34, -16), "ET": (34, -16)},
        "THR": {"XGB": (-30, -18), "ET": (30, -18)},
        "TSP": {"XGB": (-30, 16), "ET": (30, 16)},
        "Flexural_Strength": {"XGB": (-34, 14), "ET": (34, 14)},
    }

    def add_model_label(point, target, model_code, color):
        if point is None:
            return
        dx, dy = offsets[target][model_code]
        label = DISP[target].replace("Flexural", "Flex.")
        ax.annotate(
            label, xy=point, xytext=(dx, dy), textcoords="offset points",
            ha="center", va="center", fontsize=FS - 7, fontweight="bold", color=color,
            annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78),
            arrowprops=dict(arrowstyle="-", color=color, linewidth=0.60, alpha=0.65, shrinkA=2, shrinkB=3),
            zorder=7,
        )

    # Both model families labelled
    for t in TARGETS:
        add_model_label(xb_points[t], t, "XGB", COLOR_BEFORE)
        add_model_label(et_points[t], t, "ET", COLOR_AFTER)

    # State labels
    state_fs = FS - 8
    ax.text((xmin - 1) / 2, ymax - 0.035, "Destructive", ha="center", va="top", fontsize=state_fs, color=COLOR_AFTER, fontweight="bold")
    ax.text(0, ymax - 0.035, "Inert", ha="center", va="top", fontsize=state_fs, color=DARK, fontweight="bold")
    ax.text((1 + xmax) / 2, 0.46, "Stable", ha="center", va="top", fontsize=state_fs, color=COLOR_GREEN, fontweight="bold")
    ax.text((1 + xmax) / 2, ymax - 0.035, "Plastic", ha="center", va="top", fontsize=state_fs, color=COLOR_PURPLE, fontweight="bold")

    # Axes
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Normalized utility", fontweight="bold", fontsize=FS - 2)
    ax.set_ylabel("Top-10 membership plasticity" + chr(10) + "(1 - Jaccard)", fontweight="bold", fontsize=FS - 3)
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.25, zorder=0)
    ax.text(-0.08, 1.02, "(e)", transform=ax.transAxes, fontsize=FS + 4, fontweight="bold")

    # Legend above panel
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=COLOR_BEFORE, markeredgecolor="white", markersize=9, label="XGBoost"),
            Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=COLOR_AFTER, markeredgecolor="white", markersize=9, label="ExtraTrees"),
        ],
        frameon=False, fontsize=FS - 7, loc="lower center", bbox_to_anchor=(0.5, 1.005),
        ncol=2, columnspacing=1.1, handletextpad=0.35, borderaxespad=0,
    )

def p_f_utility(ax, data):
    fm = data["fm"]; sens = data["sens"]
    fm42 = fm[(fm["seed"]==42)&(fm["protocol"]=="frozen")]; s42 = sens[sens["seed"]==42]
    xb=[]; et=[]; labels=[]
    for t in TARGETS:
        xr=fm42[fm42["target"]==t]; er=s42[(s42["target"]==t)&(s42["model_family"]=="ExtraTrees")]
        if len(xr)>0 and len(er)>0:
            xb.append(xr["improvement_delta"].values[0]); et.append(er["utility"].values[0])
            labels.append(DISP[t])
    x=np.arange(len(labels)); w=0.35
    ax.bar(x-w/2,xb,w,color=COLOR_BEFORE,edgecolor="white",linewidth=0.5,label="XGBoost")
    ax.bar(x+w/2,et,w,color=COLOR_AFTER,edgecolor="white",linewidth=0.5,label="ExtraTrees")
    ax.axhline(0,color=DARK,linewidth=0.7,linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,fontsize=FS-2,fontweight="bold")
    ax.set_ylabel("Delta primary metric",fontweight="bold",fontsize=FS-2)
    ax.legend(frameon=True,fancybox=False,edgecolor="#777",fontsize=FS-6,loc="lower left")
    ax.grid(axis="y",linestyle=":",alpha=0.3)
    ax.text(-0.08,1.02,"(f)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def p_g_jaccard(ax, data):
    sim = data["sim"]; frozen = sim[sim["protocol"]=="frozen"]
    meds=[]; q1s=[]; q3s=[]; labels=[]
    for t in TARGETS:
        sub=frozen[frozen["target"]==t]; jac=sub["top10_jaccard"].dropna().values
        if len(jac)>0:
            meds.append(np.median(jac)); q1s.append(np.percentile(jac,25)); q3s.append(np.percentile(jac,75)); labels.append(DISP[t])
    x=np.arange(len(labels))
    ax.bar(x,meds,0.55,color=COLOR_BEFORE,edgecolor="white",linewidth=0.5)
    ax.errorbar(x,meds,yerr=[[m-q1 for m,q1 in zip(meds,q1s)],[q3-m for m,q3 in zip(meds,q3s)]],fmt="none",ecolor=DARK,capsize=4,linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,fontsize=FS-2,fontweight="bold")
    ax.set_ylabel("Before-after Top-10 Jaccard",fontweight="bold",fontsize=FS-2); ax.set_ylim(0,1.05)
    ax.grid(axis="y",linestyle=":",alpha=0.3)
    ax.text(-0.08,1.02,"(g)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def p_h_spearman(ax, data):
    sim = data["sim"]; frozen = sim[sim["protocol"]=="frozen"]
    meds=[]; q1s=[]; q3s=[]; labels=[]
    for t in TARGETS:
        sub=frozen[frozen["target"]==t]; rho=sub["spearman_rho"].dropna().values
        if len(rho)>0:
            meds.append(np.median(rho)); q1s.append(np.percentile(rho,25)); q3s.append(np.percentile(rho,75)); labels.append(DISP[t])
    x=np.arange(len(labels))
    ax.bar(x,meds,0.55,color=COLOR_BEFORE,edgecolor="white",linewidth=0.5)
    ax.errorbar(x,meds,yerr=[[m-q1 for m,q1 in zip(meds,q1s)],[q3-m for m,q3 in zip(meds,q3s)]],fmt="none",ecolor=DARK,capsize=4,linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,fontsize=FS-2,fontweight="bold")
    ax.set_ylabel("Before-after Spearman rho",fontweight="bold",fontsize=FS-2); ax.set_ylim(0,1.05)
    ax.grid(axis="y",linestyle=":",alpha=0.3)
    ax.text(-0.08,1.02,"(h)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def p_i_state_frequency(ax, data):
    """Cross-seed update-state frequency — dynamically computed."""
    states = compute_update_states(data)
    order = ["Inert","Stable","Destructive","Plastic"]
    colors = [COLOR_BEFORE,COLOR_GREEN,COLOR_AFTER,COLOR_PURPLE]
    y = np.arange(5)[::-1]
    cum = np.zeros(5)
    for si, state in enumerate(order):
        vals = np.array([states[t][si]/10*100 for t in TARGETS])
        ax.barh(y,vals,left=cum,color=colors[si],edgecolor="white",linewidth=0.5,label=state)
        for i in range(5):
            if vals[i]>0:
                ax.text(cum[i]+vals[i]/2,y[i],f"{int(states[TARGETS[i]][si])}",
                        ha="center",va="center",fontsize=FS-3,fontweight="bold",color="white" if vals[i]>25 else DARK)
        cum += vals
    ax.set_yticks(y); ax.set_yticklabels([DISP[t] for t in TARGETS],fontsize=FS-2,fontweight="bold")
    ax.set_xlabel("Fraction of seeds (%)",fontweight="bold",fontsize=FS-2); ax.set_xlim(0,105)
    ax.legend(frameon=True,fancybox=False,edgecolor="#777",fontsize=FS-6,loc="lower right")
    ax.text(-0.08,1.02,"(i)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

# ═══════════ MAIN ═══════════
def main():
    data = load()
    fig = plt.figure(figsize=(30, 34), dpi=300)
    gs = fig.add_gridspec(3, 3, hspace=0.28, wspace=0.32)
    axes = [fig.add_subplot(gs[r,c]) for r in range(3) for c in range(3)]
    for ax in axes: ax.set_box_aspect(1)

    p_a_zone(axes[0],data); p_b_heatmap(axes[1],data); p_c_fraction(axes[2],data)
    p_d_heatmap(axes[3],data); p_e_phase(axes[4],data); p_f_utility(axes[5],data)
    p_g_jaccard(axes[6],data); p_h_spearman(axes[7],data); p_i_state_frequency(axes[8],data)

    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SAVE,f"Figure4_5_Combined.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("[OK] Combined saved")

    # Save individual panels
    fns = [
        ("a",p_a_zone),("b",p_b_heatmap),("c",p_c_fraction),
        ("d",p_d_heatmap),("e",p_e_phase),("f",p_f_utility),
        ("g",p_g_jaccard),("h",p_h_spearman),("i",p_i_state_frequency),
    ]
    for pid, fn in fns:
        fi, axi = plt.subplots(figsize=(6,6), dpi=300)
        fn(axi, data)
        if pid != "c": axi.set_box_aspect(1)
        plt.tight_layout()
        for fmt in ["png","pdf","svg"]:
            fi.savefig(os.path.join(SAVE,f"Figure4_5_{pid}.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
        plt.close(fi)
        print(f"[OK] Panel {pid} saved")
    print(f"All -> {SAVE}")

if __name__=="__main__": main()
