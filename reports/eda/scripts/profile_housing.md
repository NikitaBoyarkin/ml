# EDA Report

**Rows:** 500 · **Columns:** 6 · **Memory:** 0.03 MB

## Profile

| Column | dtype |
|---|---|
| Id | str |
| SquareFeet | int64 |
| Bedrooms | int64 |
| HouseAge | int64 |
| State | str |
| Price | float64 |

## Data Quality

- Duplicated rows: **0**
- Constant columns: **0**
- Пропусков нет.

## Numeric Distributions

| Column | n | mean | median | std | p25 | p75 | p95 | outliers | skew |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SquareFeet | 500 | 1890.9 | 1894.5 | 622.1004 | 1463.25 | 2298.75 | 2884.2 | 1.4% | 0.0514 |
| Bedrooms | 500 | 3.578 | 4.0 | 1.5468 | 2.0 | 5.0 | 6.0 | 0.0% | 0.186 |
| HouseAge | 500 | 18.616 | 18.0 | 12.7312 | 9.0 | 28.0 | 40.05 | 0.2% | 0.3878 |
| Price | 500 | 426073.9509 | 419880.235 | 140167.9476 | 330641.4225 | 522272.45 | 645599.9755 | 0.4% | 0.0834 |

## Categorical Slices

**Id** — кардинальность 500

| Value | Count | Share |
|---|---:|---:|
| H0001 | 1 | 0.2% |
| H0002 | 1 | 0.2% |
| H0003 | 1 | 0.2% |
| H0004 | 1 | 0.2% |
| H0005 | 1 | 0.2% |
| H0006 | 1 | 0.2% |
| H0007 | 1 | 0.2% |
| H0008 | 1 | 0.2% |
| H0009 | 1 | 0.2% |
| H0010 | 1 | 0.2% |

**State** — кардинальность 3

| Value | Count | Share |
|---|---:|---:|
| New York | 172 | 34.4% |
| California | 166 | 33.2% |
| Florida | 162 | 32.4% |

## Correlations

| Pair | r | Note |
|---|---:|---|
| SquareFeet ~ Price | 0.9508 | multicollinearity |
| SquareFeet ~ Bedrooms | 0.8936 | multicollinearity |
| Bedrooms ~ Price | 0.878 | multicollinearity |

## Metric by Segment

### State (overall mean = 426073.9509, gap x1.1638)

| Value | Count | Mean | Sum | Share | vs overall |
|---|---:|---:|---:|---:|---:|
| New York | 172 | 448098.0284 | 77072860.88 | 36.2% | +5.2% |
| California | 166 | 443319.549 | 73591045.13 | 34.5% | +4.0% |
| Florida | 162 | 385018.9472 | 62373069.45 | 29.3% | -9.6% |


## Findings

- **[WARN]** Id: кардинальность 500 (100% строк) — похоже на ID, не группировать
- **[WARN]** SquareFeet ~ Price: r=0.9508 — мультиколлинеарность или дубль признака
- **[WARN]** SquareFeet ~ Bedrooms: r=0.8936 — мультиколлинеарность или дубль признака
- **[WARN]** Bedrooms ~ Price: r=0.878 — мультиколлинеарность или дубль признака

## What We Did Not Cover

- Причинность: корреляции не доказывают влияние.
- Внешние источники и бизнес-контекст, которых нет в датасете.
- Формальные тесты гипотез (это CDA / A-B testing).

## Reproducibility

```bash
uv run python .claude/skills/eda/scripts/eda_profile.py --input <dataset> \
    [--metric <col>] [--segments <a,b>] [--time-col <col>] [--top <n>]
```
