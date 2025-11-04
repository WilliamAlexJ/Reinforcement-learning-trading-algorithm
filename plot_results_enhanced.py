import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("backtests/SPY_dqn_backtest_results.csv", parse_dates=["date"]).set_index("date")

try:
    features = pd.read_csv("data/features_SPY.csv", parse_dates=["Date"]).set_index("Date")
    df = df.join(features[["signal_valid", "dist_shift_flag"]], how="left")
except Exception:
    df["signal_valid"] = 1
    df["dist_shift_flag"] = 0

#Compute buy/sell markers
# Assuming position >0 = long, <0 = short, 0 = flat
df["pos_change"] = df["position"].diff().fillna(0)
buy_signals = df[df["pos_change"] > 0]
sell_signals = df[df["pos_change"] < 0]

#Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["equity"], label="Equity", linewidth=2, color="steelblue")

# Mark buy/sell points
ax.scatter(buy_signals.index, buy_signals["equity"], marker="^", color="green", s=70, label="Buy")
ax.scatter(sell_signals.index, sell_signals["equity"], marker="v", color="red", s=70, label="Sell")

# Shade regions with distribution shifts (KS-test flagged)
shift_regions = df[df["dist_shift_flag"] == 1]
for t in shift_regions.index:
    ax.axvline(t, color="orange", alpha=0.08)

# Add title & labels
ax.set_title("RL Agent Equity Curve with Buy/Sell Signals and Regime Flags", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Equity")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
