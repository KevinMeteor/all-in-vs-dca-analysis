# 📈 All-in vs DCA: When Does Timing Matter?

## 📄 Project Deliverables

- [One-page Summary](docs/All_in_vs_DCA_one_page(English).pdf)
- [Slides](docs/All-in_vs_DCA_Analytics(Traditional Chinese).pdf)

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

---

### 3. Monte Carlo (Long-term)
- All-in wins ~65% of the time
- DCA achieves similar or slightly higher average return
- Differences persist across 10–30 year horizons

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