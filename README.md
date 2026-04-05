# 📈 All-in vs DCA: When Does Timing Matter?

## 📄 Project Deliverables

## Project Deliverables

- [One-page Summary (PDF)](docs/one_page_en.pdf)
- [Slides (PDF)](docs/slides_tc.pdf)
- [Slides (PowerPoint)](docs/slides_tc.pptx)




## 🎯 Problem

Investors often face a key decision:

- Invest all capital at once (All-in)?
- Or spread investment over time (DCA)?

This project evaluates the trade-off between:

- return maximization
- downside risk
- entry timing uncertainty

---

## ⚙️ Approach

We analyze the problem using four methods:

1. **Historical Backtest**  
   Compare yearly performance (2009–2025)

2. **Peak Entry Scenario**  
   Evaluate performance when entering near market highs



3. **Monte Carlo Simulation (GBM & Bootstrap)**  
   Simulate long-term outcomes (3–30 years)

4. **Conditional Analysis Near High**  
   Study strategy performance under unfavorable entry timing

---

## 📊  Key Results

### 1. All-in vs DCA (Overall Market)
- All-in → higher expected return
- DCA → lower downside risk

---

### 2. Peak Entry Scenario (Near Market Highs)
- DCA outperforms in ~55% of cases
- DCA reduces drawdown and tail risk
- In some cases, DCA turns losses into gains


<p align="center">
  <img src="docs/peak_dashboard.png" width="700">
</p>

This dashboard summarizes strategy performance under peak-entry scenarios.
Peak-entry analysis shows that DCA becomes more competitive under unfavorable timing conditions.


---

### 3. Monte Carlo (Long-term)
- All-in wins ~65% of the time
- DCA achieves similar or slightly higher average return
- Differences persist across 10–30 year horizons

## 🎬 Horizon Comparison: Bootstrap vs GBM

<p align="center">
  <b>Bootstrap (Historical Resampling)</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>GBM (Parametric Simulation)</b>
</p>

<p align="center">
  <img src="docs/cagr_bootstrap.gif" width="350">
  <img src="docs/cagr_gbm.gif" width="350">
</p>


- Bootstrap reflects empirical market behavior based on historical data  
- GBM assumes normally distributed returns and smooth stochastic dynamics 

This comparison highlights how modeling assumptions affect long-term return distributions.

<!-- ![GBM](docs/cagr_gbm_animation.gif)


![Bootstrap](docs/cagr_bootstrap_animation.gif) -->



---

### 4. 🧠 Key Insight

> Initial entry timing dominates long-term outcomes

Even over 30 years, relative performance is largely determined at entry.

---

## Interpretation

- All-in maximizes expected return
- DCA mitigates timing risk and reduces extreme losses

👉 Trade-off: **Return vs Timing Risk**

---

## Conclusion

- Use **All-in** when:
  - expected return is positive
  - timing risk is low

- Use **DCA** when:
  - entering near market highs
  - timing is uncertain

---

⚠️ Limitations
IID assumption in bootstrap
Normality assumption in GBM
No transaction cost
No regime modeling

---

## Future Work

- Regime-switching models  
- Multi-asset portfolios  
- Real-world constraints (costs, taxes)

---

## ⚙️ Environment Setup
Minimal dependencies are intentionally specified for reproducibility.

```bash
pip install -r requirements.txt


## 🛠 Visualization Pipeline

GIF animations are generated using a custom script:

```bash
python scripts/make_gif.py

---

