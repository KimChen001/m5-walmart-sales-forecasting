# Archived ORCD outputs

| Job | Files |
|---|---|
| [29323394](xgb-29323394) · XGBoost | 28-day and next-day exports for all baselines and M1–M3. Each directory contains full-panel and regular-seller metrics, WRMSSE by level and by window, WAPE by segment, daily totals and M3 feature importance, which [`notebooks/xgboost_full_panel.ipynb`](../notebooks/xgboost_full_panel.ipynb) reads. The job-status and log excerpt contains the final 80 lines collected from Slurm and the full-panel CSV output. |
| [27920694](timesfm3-27920694) · Sales-only TimesFM 3.0 | WAPE and 12-level WRMSSE computed from the original saved predictions. |
| [28701113](timesfm3-28701113) · Adjusted TimesFM 3.0 | Slurm completion record, full precision metrics for all four configurations, and WRMSSE for all 14 rolling test windows. |

XGBoost uses `WRMSSE` for the 12-level score and `WRMSSE_L12` for the item–store level. The sales-only TimesFM rescore and XGBoost zero-fill positions without saved predictions for WRMSSE. The adjusted TimesFM run uses the complete forecast panel. The archive contains the exported metrics and the available log excerpt. Row-level predictions and full Slurm logs remain on ORCD.
