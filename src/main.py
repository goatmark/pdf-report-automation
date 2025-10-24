# -*- coding: utf-8 -*-
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from google.cloud import bigquery
from google.oauth2 import service_account
import dotenv
from dotenv import load_dotenv

load_dotenv()

# ---- project setup ----
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)

# service-account key comes from env var GCP_SA_KEY_JSON
gcp_key_json = os.environ.get("GCP_SA_KEY_JSON")
if not gcp_key_json:
    raise RuntimeError("missing env var GCP_SA_KEY_JSON")

key_path = Path("/tmp/key.json")
key_path.write_text(gcp_key_json)  # write the JSON string to a file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path)

creds = service_account.Credentials.from_service_account_file(str(key_path))
project_id = creds.project_id
client = bigquery.Client(project=project_id, credentials=creds)

# ---- first graphs (left intact; you can ignore them) ----
query_hist = f"""
select
  date,
  duration_min,
  distance_mi,
  calories,
  (calories / duration_min * 60.0) as calories_per_hour,
  (distance_mi / duration_min * 60.0) as mph
from `{project_id}.dbt_mkhoury.cln_exercise_log`
where exercise_label = 'Treadmill'
  and (distance_mi / duration_min * 60.0) > 5
  and distance_mi = 1
order by date
"""
df = client.query(query_hist).result().to_dataframe()
df["mph"] = pd.to_numeric(df["mph"], errors="coerce").astype(float)
df["calories_per_hour"] = pd.to_numeric(df["calories_per_hour"], errors="coerce").astype(float)

bin_size = 0.2
mph = df["mph"].dropna().to_numpy(dtype=float)
if mph.size > 0:
    low = np.floor(mph.min() / bin_size) * bin_size
    high = np.ceil(mph.max() / bin_size) * bin_size + bin_size
    bins = np.arange(low, high + 1e-9, bin_size)

    mean = float(np.mean(mph))
    std = float(np.std(mph, ddof=1)) if mph.size > 1 else 0.0
    last_mph = float(mph[-1])
    percentile = float((mph <= last_mph).mean() * 100.0)

    counts, edges = np.histogram(mph, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n = mph.size
    bw = bin_size

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=140)
    ax.bar(centers, counts, width=bw * 0.95)
    if std > 0:
        x = np.linspace(edges[0], edges[-1], 500)
        dens = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        ax.plot(x, dens * n * bw, linewidth=2.0)
    ax.axvline(last_mph, linestyle="--", linewidth=1.5)
    ax.set_xlabel("mph", labelpad=8)
    ax.set_ylabel("count", labelpad=8)
    ax.set_title(f"mean: {mean:.2f} mph · last: {last_mph:.2f} mph · {percentile:.1f} percentile", pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "treadmill_hist.png", bbox_inches="tight")
    plt.close(fig)

query_scatter = f"""
select
  date,
  duration_min,
  distance_mi,
  (distance_mi / duration_min * 60.0) as mph
from `{project_id}.dbt_mkhoury.cln_exercise_log`
where exercise_label = 'Treadmill'
  and (distance_mi / duration_min * 60.0) > 5
order by date
"""
scatter_df = client.query(query_scatter).result().to_dataframe()
scatter_df["mph"] = pd.to_numeric(scatter_df["mph"], errors="coerce").astype(float)
scatter_df["duration_min"] = pd.to_numeric(scatter_df["duration_min"], errors="coerce").astype(float)
sc = scatter_df[scatter_df["duration_min"] > 0].dropna(subset=["mph", "duration_min"]).copy()

if not sc.empty:
    x = sc["mph"].to_numpy(dtype=float)
    y = sc["duration_min"].to_numpy(dtype=float)
    w = y.astype(float)
    lo, hi = np.percentile(w, [5, 95])
    w = np.clip(w, lo, hi)
    w = w / np.mean(w)
    coef = np.polyfit(x, y, deg=2, w=w)
    poly = np.poly1d(coef)
    xs = np.linspace(x.min(), x.max(), 600)
    ys = poly(xs)
    resid = y - poly(x)

    def weighted_quantile(vals, q, weights):
        sorter = np.argsort(vals)
        v = vals[sorter]
        wts = weights[sorter]
        cw = np.cumsum(wts)
        cw = (cw - 0.5 * wts) / np.sum(wts)
        return np.interp(q, cw, v)

    q10 = float(weighted_quantile(resid, 0.10, w))
    q90 = float(weighted_quantile(resid, 0.90, w))
    band_lo = ys + q10
    band_hi = ys + q90
    is_low = resid < q10
    is_high = resid > q90

    fig2, ax2 = plt.subplots(figsize=(9.6, 5.8), dpi=140)
    ax2.fill_between(xs, band_lo, band_hi, alpha=0.18)
    ax2.plot(xs, ys, linewidth=2.6)
    ax2.scatter(x, y, s=34, alpha=0.85, edgecolors="white", linewidths=0.4)
    if is_high.any():
        ax2.scatter(x[is_high], y[is_high], s=52, facecolors="none", edgecolors="black", linewidths=1.2)
    if is_low.any():
        ax2.scatter(x[is_low], y[is_low], s=52, facecolors="none", edgecolors="black", linewidths=1.2)
    ax2.set_xlabel("mph", labelpad=8)
    ax2.set_ylabel("duration (min)", labelpad=8)
    ax2.set_title("treadmill: duration vs mph · quadratic frontier with weighted 10–90% band", pad=12)
    ax2.grid(alpha=0.28)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(OUT / "treadmill_scatter.png", bbox_inches="tight")
    plt.close(fig2)

# ---- monthly aggregates for report ----
monthly_sql = f"""
with monthly as (
  select
    date_trunc(date, month) as month_date,
    coalesce(sum(total_workouts), 0) as total_workouts,
    coalesce(sum(total_runs), 0) as total_runs,
    coalesce(sum(total_dishes), 0) as total_dishes,
    coalesce(sum(total_revenue), 0) as total_revenue,
    coalesce(sum(total_spend), 0) as total_spend
  from `{project_id}.dbt_mkhoury.rpt_metrics_daily`
  where date >= date '2024-01-01'
  group by 1
)
select *
from monthly
order by month_date
"""
monthly_df = client.query(monthly_sql).result().to_dataframe()
if monthly_df.empty:
    raise ValueError("rpt_metrics_daily returned no rows")

monthly_df["month_date"] = pd.to_datetime(monthly_df["month_date"], format="%Y-%m-%d")
monthly_df["month_label"] = monthly_df["month_date"].dt.to_period("M").astype(str)
metrics = ["total_workouts", "total_runs", "total_dishes", "total_revenue", "total_spend"]
for m in metrics:
    monthly_df[m] = pd.to_numeric(monthly_df[m], errors="coerce").fillna(0.0)

# charts -> png
chart_entries = []
for m in metrics:
    figm, axm = plt.subplots(figsize=(9, 3), dpi=140)
    axm.bar(monthly_df["month_label"], monthly_df[m])
    axm.set_title(m.replace("_", " "), fontsize=11)
    axm.set_ylabel("value", fontsize=9)
    axm.tick_params(axis="x", labelrotation=45)
    axm.grid(axis="y", alpha=0.3)
    axm.spines["top"].set_visible(False)
    axm.spines["right"].set_visible(False)
    axm.margins(x=0.01)
    figm.tight_layout()
    fname = f"{m}.png"
    figm.savefig(OUT / fname, bbox_inches="tight")
    plt.close(figm)
    chart_entries.append({"title": m.replace("_", " "), "src": fname})

# ---- render pdf with weasyprint ----
env = Environment(
    loader=FileSystemLoader(str(SRC / "templates")),
    autoescape=select_autoescape()
)
template = env.get_template("report.html")

date_range = f"{monthly_df['month_label'].iloc[0]} — {monthly_df['month_label'].iloc[-1]}"
html_str = template.render(
    title="monthly metrics",
    subtitle=date_range,
    generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    items=chart_entries,
    css_path="../styles/report.css",
)

pdf_path = OUT / "monthly_report.pdf"
HTML(string=html_str, base_url=str(OUT)).write_pdf(str(pdf_path))

print(f"wrote pdf: {pdf_path}")
