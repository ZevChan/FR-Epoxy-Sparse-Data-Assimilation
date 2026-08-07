"""
plot_figure4.py — Figure 4: Spatial Error Redistribution (2x2, pure spatial)
================================================================================
(a) Zone definition: distance distribution + tertile boundaries per target
(b) 5 targets x 3 zones relative loss improvement heatmap
(c) Fraction improved per zone
(d) Distance vs delta-abs-error binned trend

All results are within the fixed literature holdout domain.
Model-family panels (ExtraTrees) moved to Figure 5.
"""
import os, warnings
BASE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

KA = os.path.join(BASE, "Results", "Knowledge_Assimilation")
SAVE = os.path.join(BASE, "Graphs", "Figure4_Spatial")
os.makedirs(SAVE, exist_ok=True)

COLOR_BEFORE = "#5DA5DA"; COLOR_AFTER = "#C91511"; GRAY = "#8C92AC"; DARK = "#333333"
FS = 14
plt.rcParams.update({"font.family":"Arial","font.size":FS,"axes.unicode_minus":False,
    "axes.linewidth":0.7,"xtick.major.width":0.7,"ytick.major.width":0.7,
    "xtick.direction":"in","ytick.direction":"in","xtick.major.size":3,"ytick.major.size":3,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})

TARGETS = ["LOI","UL94_Rating","THR","TSP","Flexural_Strength"]
DISP = {"LOI":"LOI","UL94_Rating":"UL-94","THR":"THR","TSP":"TSP","Flexural_Strength":"Flexural"}
ZONE_COLORS = {"near":COLOR_BEFORE,"middle":GRAY,"far":COLOR_AFTER}

def load():
    d = {}
    d["regions"] = pd.read_csv(os.path.join(KA,"Propagation","propagation_regions.csv"))
    d["samples"] = pd.read_csv(os.path.join(KA,"Propagation","propagation_samples.csv"))
    return d

def panel_a_zone_definition(ax, data):
    """Distance distribution + tertile boundaries per target (seed 42)."""
    samples = data["samples"]; s42 = samples[samples["seed"]==42]
    # Get tertile boundaries from regions
    regions = data["regions"]; r42 = regions[regions["seed"]==42]
    for i, t in enumerate(TARGETS):
        sub = s42[s42["target"]==t]
        dists = sub["distance_to_nearest_experiment"].dropna().values
        if len(dists)==0: continue
        # KDE curve
        from scipy.stats import gaussian_kde
        xs = np.linspace(dists.min(), dists.max(), 200)
        try:
            kde = gaussian_kde(dists)
            ys = kde(xs); ys = ys/ys.max()*0.4
            ax.plot(xs, ys+i, color=DARK, linewidth=1.2, alpha=0.7)
        except: pass
        # Tertile boundaries
        sub_r = r42[r42["target"]==t]
        if len(sub_r)>0:
            d_near = sub_r[sub_r["neighborhood"]=="near"]["distance_max"].values
            d_mid = sub_r[sub_r["neighborhood"]=="middle"]["distance_max"].values
            if len(d_near)>0:
                ax.axvline(d_near[0], ymin=(i-0.3)/len(TARGETS), ymax=(i+0.3)/len(TARGETS),
                          color=COLOR_BEFORE, linewidth=1.5, linestyle="--", alpha=0.6)
            if len(d_mid)>0:
                ax.axvline(d_mid[0], ymin=(i-0.3)/len(TARGETS), ymax=(i+0.3)/len(TARGETS),
                          color=COLOR_AFTER, linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(TARGETS)))
    ax.set_yticklabels([DISP[t] for t in TARGETS], fontsize=FS-1, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Distance to nearest experiment", fontweight="bold")
    ax.set_title("Near / Middle / Far tertile boundaries", fontsize=FS-1, fontweight="bold", color=DARK)
    # Legend
    leg = [Line2D([0],[0],color=COLOR_BEFORE,linestyle="--",lw=1.5,label="Near/Mid boundary"),
           Line2D([0],[0],color=COLOR_AFTER,linestyle="--",lw=1.5,label="Mid/Far boundary")]
    ax.legend(handles=leg, frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-4, loc="lower right")
    ax.text(-0.08,1.02,"(a)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def panel_b_heatmap(ax, data):
    """5 targets x 3 zones: relative loss improvement heatmap."""
    regions = data["regions"]; r42 = regions[regions["seed"]==42]
    zones = ["near","middle","far"]
    mat = np.zeros((len(TARGETS),3))
    annot = []
    for i, t in enumerate(TARGETS):
        row_annot = []
        for j, z in enumerate(zones):
            sub = r42[(r42["target"]==t)&(r42["neighborhood"]==z)]
            if len(sub)>0:
                # Relative loss improvement: (before-after)/before, positive=improvement
                b = sub["mean_loss_before"].values[0]; a = sub["mean_loss_after"].values[0]
                rel = (b-a)/b if abs(b)>1e-8 else 0
                mat[i,j] = rel
                row_annot.append(f"{rel:+.3f}")
            else:
                row_annot.append("")
        annot.append(row_annot)
    vmax = max(abs(np.min(mat)), abs(np.max(mat))) if np.max(np.abs(mat))>0 else 0.01
    from matplotlib.colors import LinearSegmentedColormap
    cmap_b = LinearSegmentedColormap.from_list("rg_b", [COLOR_AFTER, "#E8E8E8", COLOR_BEFORE])
    im = ax.imshow(mat, cmap=cmap_b, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Near","Middle","Far"], fontsize=FS-1, fontweight="bold")
    ax.set_yticks(range(len(TARGETS))); ax.set_yticklabels([DISP[t] for t in TARGETS], fontsize=FS-1, fontweight="bold")
    for i in range(len(TARGETS)):
        for j in range(3):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=FS-1, fontweight="bold",
                    color="white" if abs(mat[i,j])>np.max(np.abs(mat))*0.5 else "black")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Rel. loss improvement", fontsize=FS-2)
    ax.text(-0.08,1.02,"(b)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def panel_c_fraction_improved(ax, data):
    """Fraction of test samples improved per zone per target (broken y-axis)."""
    regions = data["regions"]; r42 = regions[regions["seed"]==42]
    zones = ["near","middle","far"]; x = np.arange(len(TARGETS)); w = 0.22

    # Compute values first
    zone_vals = {}
    for j, z in enumerate(zones):
        vals = []
        for i, t in enumerate(TARGETS):
            sub = r42[(r42["target"]==t)&(r42["neighborhood"]==z)]
            vals.append(sub["fraction_improved"].values[0] if len(sub)>0 else 0)
        zone_vals[z] = (list(range(len(TARGETS))), vals)

    # Create two sub-axes for broken y-axis
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # Get original ax position
    bbox = ax.get_position()
    # Manual: create top (0.35–0.80) and bottom (0–0.15) axes
    fig = ax.figure
    ax.remove()
    # Bottom axis: 0–0.15, takes 30% height, at bottom of original bbox
    h = bbox.height; y0 = bbox.y0
    ax_bot = fig.add_axes([bbox.x0, y0, bbox.width, h*0.18])
    ax_top = fig.add_axes([bbox.x0, y0+h*0.25, bbox.width, h*0.75])

    for ax_i in [ax_bot, ax_top]:
        for j, z in enumerate(zones):
            all_pos, vals = zone_vals[z]
            ax_i.bar(np.array(all_pos)+w*(j-1), vals, w, color=ZONE_COLORS[z], edgecolor="white", linewidth=0.4)
        ax_i.axhline(0.5, color=DARK, linewidth=0.7, linestyle="--", alpha=0.5)
        ax_i.set_xticks(x); ax_i.set_xticklabels([DISP[t] for t in TARGETS], rotation=20, fontsize=FS-1, fontweight="bold")
        ax_i.grid(axis="y", linestyle=":", alpha=0.3)
    ax_top.annotate("50% of samples improved", xy=(4.5,0.51), fontsize=FS-4, color=DARK, fontstyle="italic", ha="right")

    ax_bot.set_ylim(0, 0.05); ax_top.set_ylim(0.35, 0.70)
    # Hide top spines on bottom, bottom spines on top
    ax_bot.spines["top"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    # Remove x labels from top
    ax_top.set_xticklabels([]); ax_top.set_xlabel("")
    # Break marks (diagonal lines on the upper/lower edges)
    d = 0.02
    kw = dict(transform=ax_bot.transAxes, color=DARK, clip_on=False, linewidth=1.2)
    ax_bot.plot((-d, +d), (1-2*d, 1), **kw)
    ax_bot.plot((-d, +d), (1-2*d, 1), **kw)
    kw = dict(transform=ax_top.transAxes, color=DARK, clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-0.02, 0.02), **kw)
    ax_top.plot((-d, +d), (-0.02, 0.02), **kw)
    # Shared y label
    fig.text(bbox.x0-0.06, bbox.y0+bbox.height/2, "Fraction improved", ha="center", va="center",
             rotation=90, fontweight="bold", fontsize=FS)
    # Legend on top axis
    ax_top.legend(handles=[Patch(facecolor=COLOR_BEFORE,edgecolor="white",label="Near"),
                   Patch(facecolor=GRAY,edgecolor="white",label="Middle"),
                   Patch(facecolor=COLOR_AFTER,edgecolor="white",label="Far")],
                  frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-4, loc="lower left")
    # Panel label
    ax_top.text(-0.10, 1.02, "(c)", transform=ax_top.transAxes, fontsize=FS+4, fontweight="bold")

def panel_d_distance_scatter(ax, data):
    """Distance vs delta abs-error: binned median + IQR (no raw scatter)."""
    samples = data["samples"]; s42 = samples[samples["seed"]==42]
    reg_targets = ["LOI","THR","TSP","Flexural_Strength"]
    for i, t in enumerate(reg_targets):
        sub = s42[s42["target"]==t]
        dist = sub["distance_percentile"].dropna().values
        delta = (sub["abs_error_before"] - sub["abs_error_after"]).dropna().values
        c = plt.cm.tab10(i)
        # Binned median + IQR
        bins = np.percentile(dist, [0,20,40,60,80,100])
        bin_medians = []; bin_q1s = []; bin_q3s = []; bin_centers = []
        for bi in range(len(bins)-1):
            mask = (dist>=bins[bi])&(dist<bins[bi+1])
            if mask.sum()>3:
                bin_medians.append(np.median(delta[mask]))
                bin_q1s.append(np.percentile(delta[mask],25))
                bin_q3s.append(np.percentile(delta[mask],75))
                bin_centers.append((bins[bi]+bins[bi+1])/2)
        if len(bin_centers)>1:
            ax.fill_between(bin_centers, bin_q1s, bin_q3s, color=c, alpha=0.15)
            ax.plot(bin_centers, bin_medians, color=c, linewidth=2.5, marker='o', markersize=6, label=DISP[t])
    ax.axhline(0, color=DARK, linewidth=0.7, linestyle="--")
    ax.set_xlabel("Distance percentile to nearest experiment", fontweight="bold")
    ax.set_ylabel("Delta |Error| (Before - After)", fontweight="bold")
    ax.legend(frameon=True, fancybox=False, edgecolor="#777", fontsize=FS-4, loc="lower left", ncol=2, title="Binned median + IQR")
    ax.grid(linestyle=":", alpha=0.3)
    ax.text(-0.08,1.02,"(d)",transform=ax.transAxes,fontsize=FS+4,fontweight="bold")

def main():
    data = load()
    fig = plt.figure(figsize=(20, 20), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(2)]
    for ax in axes: ax.set_box_aspect(1)

    panel_a_zone_definition(axes[0], data); panel_b_heatmap(axes[1], data)
    panel_c_fraction_improved(axes[2], data); panel_d_distance_scatter(axes[3], data)

    for fmt in ["png","pdf","svg"]:
        fig.savefig(os.path.join(SAVE,f"Figure4.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig); print("[OK] Figure4 saved")

    panels = [("a",panel_a_zone_definition),("b",panel_b_heatmap),
              ("c",panel_c_fraction_improved),("d",panel_d_distance_scatter)]
    for pid, fn in panels:
        fi, axi = plt.subplots(figsize=(7,7),dpi=300); fn(axi,data); axi.set_box_aspect(1)
        plt.tight_layout()
        for fmt in ["png","pdf","svg"]:
            fi.savefig(os.path.join(SAVE,f"Figure4_{pid}.{fmt}"),dpi=300,bbox_inches="tight",facecolor="white")
        plt.close(fi)
    print(f"All -> {SAVE}")

if __name__=="__main__": main()
