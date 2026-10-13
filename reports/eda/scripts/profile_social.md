# EDA Report

**Rows:** 400 · **Columns:** 5 · **Memory:** 0.02 MB

## Profile

| Column | dtype |
|---|---|
| User ID | int64 |
| Gender | str |
| Age | float64 |
| EstimatedSalary | float64 |
| Purchased | int64 |

## Data Quality

- Duplicated rows: **0**
- Constant columns: **0**
- Пропусков нет.

## Numeric Distributions

| Column | n | mean | median | std | p25 | p75 | p95 | outliers | skew |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| User ID | 400 | 15691539.7575 | 15694341.5 | 71658.3216 | 15626763.75 | 15750363.0 | 15801251.45 | 0.0% | -0.0345 |
| Age | 400 | 37.655 | 37.0 | 10.4829 | 29.75 | 46.0 | 57.05 | 0.0% | 0.2313 |
| EstimatedSalary | 400 | 69742.5 | 70000.0 | 34096.9603 | 43000.0 | 88000.0 | 137050.0 | 0.0% | 0.495 |
| Purchased | 400 | 0.3575 | 0.0 | 0.4799 | 0.0 | 1.0 | 1.0 | 0.0% | 0.5969 |

## Categorical Slices

**Gender** — кардинальность 2

| Value | Count | Share |
|---|---:|---:|
| Female | 204 | 51.0% |
| Male | 196 | 49.0% |

## Correlations

| Pair | r | Note |
|---|---:|---|
| Age ~ Purchased | 0.6225 | moderate |

## Metric by Segment

### Gender (overall mean = 0.3575, gap x1.1212)

| Value | Count | Mean | Sum | Share | vs overall |
|---|---:|---:|---:|---:|---:|
| Female | 204 | 0.3775 | 77.0 | 53.8% | +5.6% |
| Male | 196 | 0.3367 | 66.0 | 46.2% | -5.8% |


## Findings

- Критичных флагов не найдено.

## What We Did Not Cover

- Причинность: корреляции не доказывают влияние.
- Внешние источники и бизнес-контекст, которых нет в датасете.
- Формальные тесты гипотез (это CDA / A-B testing).

## Reproducibility

```bash
uv run python .claude/skills/eda/scripts/eda_profile.py --input <dataset> \
    [--metric <col>] [--segments <a,b>] [--time-col <col>] [--top <n>]
```
