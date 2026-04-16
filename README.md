# Decision Tree Analysis
### Spam Classification & Suicide Rate Prediction

Two end-to-end predictive modeling projects built in R, exploring how decision trees behave under different pruning strategies — and what that means for real-world model selection.

---

## Projects

### 1. Spam Classification
**Dataset:** HP Labs spam dataset (4,601 emails, 57 features)

Built a full classification tree on an 80/20 train-test split, then applied two pruning approaches to reduce overfitting:

- **Minimum CV error tree** — selects the depth that minimizes cross-validation error
- **1-SE rule tree (`opttree`)** — selects the simplest tree whose CV error stays within one standard deviation of the minimum

The pruned model achieved a **lower false positive rate** (fewer legitimate emails flagged as spam) at the cost of a slightly higher false negative rate — a meaningful trade-off in any production email filter where false alarms damage user trust.

| Model | Test Error | False Positive Rate | False Negative Rate |
|---|---|---|---|
| Full tree (252 leaves) | 8.47% | 6.90% | 10.53% |
| `opttree` — 1-SE rule (37 leaves) | **8.36%** | **5.94%** | 11.53% |

---

### 2. Suicide Rate Prediction
**Dataset:** Global suicide rates (multi-country, 1985–2016)

Built a regression tree to predict suicides per 100k population. Key finding: a handful of variables dominate the prediction almost entirely.

**Variable importance (relative):**

| Variable | Importance |
|---|---|
| GDP for year | 3,886,717 |
| Age group | 2,472,647 |
| Country | 2,169,463 |
| Sex | 1,623,498 |
| Generation | 1,299,269 |
| Suicides (absolute count) | 1,227,445 |
| Year | 46,003 |

GDP and age group together account for the largest share of predictive power — but what's notable is how far ahead they are from variables like `year`, which carries almost no signal. Country still matters significantly even after controlling for GDP, suggesting that structural and cultural factors create real variation that economic indicators alone don't capture.

- **CV error vs. test error:** CV error (0.208, relative) and test MSE (68.2) are on different scales, but both point to a model that generalizes reasonably well given the dataset's size and heterogeneity.

---

## What I Was Exploring

Both projects come down to the same question: how much should a model be allowed to grow?

A fully grown tree fits the training data almost perfectly — and performs worse on anything new. Heavy pruning fixes that, but at the cost of missing real structure in the data. The 1-SE rule sits in between: take the simplest tree that's still statistically competitive with the best one.

In the spam case, that trade-off had a concrete interpretation — `opttree` flagged fewer legitimate emails as spam, which in a real system matters more than a marginal improvement in overall accuracy.

---

## Stack
- **Language:** R
- **Packages:** `rpart`, `kernlab`
- **Methods:** CART, cost-complexity pruning, k-fold cross-validation, 1-SE rule

---

## Files
| File | Description |
|---|---|
| `analysis.ipynb` | Full analysis with code, outputs, and commentary |
| `data/suicide-rate.csv` | Suicide rates dataset |
