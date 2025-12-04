import discord
from discord.ext import commands
from datetime import datetime, timedelta
import csv
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from zoneinfo import ZoneInfo

MODEL_ID = "Franklin001/sentimental"
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load model once
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()


# -------- TIME PARSER --------
def parse_time_input(time_str: str):
    time_str = time_str.lower().strip()

    minute_patterns = [r"^(\d+)\s*m$", r"^(\d+)\s*mn$", r"^(\d+)\s*min$", r"^(\d+)\s*mins$", r"^(\d+)\s*minute$", r"^(\d+)\s*minutes$"]
    for p in minute_patterns:
        m = re.match(p, time_str)
        if m: return int(m.group(1))

    hour_patterns = [r"^(\d+)\s*h$", r"^(\d+)\s*hr$", r"^(\d+)\s*hrs$", r"^(\d+)\s*hour$", r"^(\d+)\s*hours$"]
    for p in hour_patterns:
        m = re.match(p, time_str)
        if m: return int(m.group(1)) * 60

    day_patterns = [r"^(\d+)\s*d$", r"^(\d+)\s*day$", r"^(\d+)\s*days$"]
    for p in day_patterns:
        m = re.match(p, time_str)
        if m: return int(m.group(1)) * 24 * 60

    return None


def format_time(minutes: int):
    hours = minutes // 60
    mins = minutes % 60
    parts = []
    if hours > 0: parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if mins > 0:  parts.append(f"{mins} minute" + ("s" if mins != 1 else ""))
    return " ".join(parts) if parts else "0 minutes"


# -------- FAST BATCH SENTIMENT --------
def analyze_batch(texts):
    """Run sentiment for all messages in ONE batch (much faster)."""
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    confidences, labels = torch.max(probs, dim=1)

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    results = []
    for label, conf in zip(labels.tolist(), confidences.tolist()):
        results.append((label_map[label], conf))

    return results


def generate_thick_bar(percentage, length=10):
    filled_units = int(round(length * percentage / 100))
    empty_units = length - filled_units
    return "▰" * filled_units + "▱" * empty_units


# ===========================================================
# MAIN COG
# ===========================================================
class Sentiment(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.local_tz = ZoneInfo("Asia/Phnom_Penh")

    @commands.command()
    async def analyze(self, ctx, *, time_input: str = "24h"):
        minutes = parse_time_input(time_input)
        if minutes is None:
            return await ctx.send("❌ Invalid time format. Example: `5m`, `2h`, `1d`")

        if minutes < 5:
            return await ctx.send("⚠️ Minimum allowed is **5 minutes**.")

        if minutes > 10080:  # 7 days
            return await ctx.send("⚠️ Maximum allowed is **7 days**.")

        now_local = datetime.now(self.local_tz)
        since_local = now_local - timedelta(minutes=minutes)
        friendly_time = format_time(minutes)

        await ctx.send(f"🕒 Fetching messages from the last **{friendly_time}**...")

        # -------- FAST HISTORY FETCH --------
        messages = []
        async for msg in ctx.channel.history(limit=2000, oldest_first=False):
            local_time = msg.created_at.replace(tzinfo=ZoneInfo("UTC")).astimezone(self.local_tz)

            if local_time < since_local:
                break  # stop early (HUGE SPEED BOOST)

            if msg.author.bot or msg.content.startswith("!"):
                continue

            messages.append([local_time.isoformat(), msg.author.name, msg.content])

        if not messages:
            return await ctx.send("⚠️ No messages found in that time range.")

        # -------- BATCH MODEL INFERENCE --------
        texts = [m[2] for m in messages]
        batch_results = analyze_batch(texts)

        full_results = []
        for (time_str, author, text), (sentiment, conf) in zip(messages, batch_results):
            full_results.append((time_str, author, text, sentiment, conf))

        # -------- SUMMARY --------
        total = len(full_results)
        pos = sum(1 for x in full_results if x[3] == "Positive")
        neu = sum(1 for x in full_results if x[3] == "Neutral")
        neg = sum(1 for x in full_results if x[3] == "Negative")

        avg_conf = sum(x[4] for x in full_results) / total * 100

        pos_pct = pos / total * 100
        neu_pct = neu / total * 100
        neg_pct = neg / total * 100

        # plural
        pos_word = "message" if pos == 1 else "messages"
        neu_word = "message" if neu == 1 else "messages"
        neg_word = "message" if neg == 1 else "messages"

        max_count_len = max(len(str(pos)), len(str(neu)), len(str(neg)))
        word_width = len("messages")
        fmt = f"{{count:>{max_count_len}}} {{word:<{word_width}}}"

        lines = [
            f"📊 Sentiment Summary (Last {friendly_time})\n",
            f"🟢 Positive  {fmt.format(count=pos, word=pos_word)}  │  {generate_thick_bar(pos_pct)}  {pos_pct:5.1f}%",
            f"⚪ Neutral   {fmt.format(count=neu, word=neu_word)}  │  {generate_thick_bar(neu_pct)}  {neu_pct:5.1f}%",
            f"🔴 Negative  {fmt.format(count=neg, word=neg_word)}  │  {generate_thick_bar(neg_pct)}  {neg_pct:5.1f}%",
            f"\n📦 Total Messages : {total}",
            f"🎯 Avg Confidence : {avg_conf:.2f}%"
        ]

        await ctx.send("```\n" + "\n".join(lines) + "\n```")

        # -------- SAVE CSV --------
        csv_file = os.path.join(DATA_DIR, "analyzed_messages.csv")
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "author", "message", "sentiment", "confidence"])
            writer.writerows(full_results)

        await ctx.send("✅ Sentiment analysis complete.")


async def setup(bot):
    await bot.add_cog(Sentiment(bot))
