# EDA Report

**Rows:** 800 · **Columns:** 9 · **Memory:** 0.07 MB

## Profile

| Column | dtype |
|---|---|
| CustomerID | int64 |
| Gender | str |
| Age | int64 |
| TenureMonths | int64 |
| Contract | str |
| InternetService | str |
| MonthlyCharges | int64 |
| TotalCharges | int64 |
| Churn | int64 |

## Data Quality

- Duplicated rows: **0**
- Constant columns: **0**
- Пропусков нет.

## Numeric Distributions

| Column | n | mean | median | std | p25 | p75 | p95 | outliers | skew |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CustomerID | 800 | 400.5 | 400.5 | 231.0844 | 200.75 | 600.25 | 760.05 | 0.0% | 0.0 |
| Age | 800 | 45.4462 | 45.0 | 16.7788 | 31.0 | 60.0 | 72.0 | 0.0% | 0.0512 |
| TenureMonths | 800 | 25.595 | 25.0 | 18.4467 | 11.0 | 38.0 | 57.0 | 0.8% | 0.4858 |
| MonthlyCharges | 800 | 70.76 | 71.0 | 17.1931 | 59.0 | 83.0 | 99.0 | 0.5% | -0.0601 |
| TotalCharges | 800 | 1886.1 | 1726.5 | 1395.5254 | 754.5 | 2841.0 | 4418.1 | 0.6% | 0.6867 |
| Churn | 800 | 0.125 | 0.0 | 0.3309 | 0.0 | 0.0 | 1.0 | 0.0% | 2.272 |

## Categorical Slices

**Contract** — кардинальность 3

| Value | Count | Share |
|---|---:|---:|
| Month-to-month | 432 | 54.0% |
| One year | 223 | 27.9% |
| Two year | 145 | 18.1% |

**InternetService** — кардинальность 3

| Value | Count | Share |
|---|---:|---:|
| Fiber optic | 326 | 40.8% |
| DSL | 263 | 32.9% |
| No | 211 | 26.4% |

**Gender** — кардинальность 2

| Value | Count | Share |
|---|---:|---:|
| Male | 416 | 52.0% |
| Female | 384 | 48.0% |

## Correlations

| Pair | r | Note |
|---|---:|---|
| TenureMonths ~ TotalCharges | 0.9222 | multicollinearity |

## Metric by Segment

### Contract (overall mean = 0.125, gap x15.7681)

| Value | Count | Mean | Sum | Share | vs overall |
|---|---:|---:|---:|---:|---:|
| Month-to-month | 432 | 0.2176 | 94.0 | 94.0% | +74.1% |
| One year | 223 | 0.0179 | 4.0 | 4.0% | -85.7% |
| Two year | 145 | 0.0138 | 2.0 | 2.0% | -89.0% |

### InternetService (overall mean = 0.125, gap x1.6022)

| Value | Count | Mean | Sum | Share | vs overall |
|---|---:|---:|---:|---:|---:|
| Fiber optic | 326 | 0.1442 | 47.0 | 47.0% | +15.3% |
| DSL | 263 | 0.1293 | 34.0 | 34.0% | +3.4% |
| No | 211 | 0.09 | 19.0 | 19.0% | -28.0% |

### Gender (overall mean = 0.125, gap x1.5578)

| Value | Count | Mean | Sum | Share | vs overall |
|---|---:|---:|---:|---:|---:|
| Female | 384 | 0.1536 | 59.0 | 59.0% | +22.9% |
| Male | 416 | 0.0986 | 41.0 | 41.0% | -21.1% |


## Findings

- **[INFO]** Churn: правосторонний скос (2.272) — медиана надёжнее среднего
- **[WARN]** TenureMonths ~ TotalCharges: r=0.9222 — мультиколлинеарность или дубль признака
- **[INFO]** Разрыв по Contract: x15.7681 между сегментами — кандидат в главный вывод
- **[INFO]** Разрыв по InternetService: x1.6022 между сегментами — кандидат в главный вывод
- **[INFO]** Разрыв по Gender: x1.5578 между сегментами — кандидат в главный вывод

## What We Did Not Cover

- Причинность: корреляции не доказывают влияние.
- Внешние источники и бизнес-контекст, которых нет в датасете.
- Формальные тесты гипотез (это CDA / A-B testing).

## Reproducibility

```bash
uv run python .claude/skills/eda/scripts/eda_profile.py --input <dataset> \
    [--metric <col>] [--segments <a,b>] [--time-col <col>] [--top <n>]
```
