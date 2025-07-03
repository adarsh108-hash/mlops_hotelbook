from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def log_results(occ_class, rew_class, final_price, occ_metrics, rew_metrics, price_metrics):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"""
🕒 Timestamp: {timestamp}
----------------------------------------
✅ Occupancy Class: {occ_class}
✅ Points Class: {rew_class}
✅ Final Room Price: ${final_price:.2f}

📊 Occupancy Metrics: {occ_metrics}
📊 Rewards Accuracy: {rew_metrics['accuracy']}
📊 Pricing Metrics: {price_metrics}

========================================
"""

    # ✅ Encode file in UTF-8 to avoid UnicodeEncodeError on Windows
    with open(LOG_DIR / "pipeline_log.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)
