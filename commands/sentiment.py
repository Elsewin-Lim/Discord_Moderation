import discord
from discord.ext import commands
from datetime import datetime, timedelta
import csv
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

MODEL_ID = "Franklin001/sentimental"
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load sentiment model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

# -------- TIME PARSER & FORMATTER --------
def parse_time_input(time_str: str):
    """
    Accepts formats like:
    5m, 5mn, 5 min, 5 minutes
    2h, 2hr, 2 hours
    1d, 2 days
    Returns total minutes as integer.
    """
    time_str = time_str.lower().strip()

    # --- Minutes ---
    minute_patterns = [
        r"^(\d+)\s*m$", r"^(\d+)\s*mn$", r"^(\d+)\s*min$",
        r"^(\d+)\s*mins$", r"^(\d+)\s*minute$", r"^(\d+)\s*minutes$"
    ]
    for p in minute_patterns:
        match = re.match(p, time_str)
        if match:
            return int(match.group(1))

    # --- Hours ---
    hour_patterns = [
        r"^(\d+)\s*h$", r"^(\d+)\s*hr$", r"^(\d+)\s*hrs$",
        r"^(\d+)\s*hour$", r"^(\d+)\s*hours$"
    ]
    for p in hour_patterns:
        match = re.match(p, time_str)
        if match:
            return int(match.group(1)) * 60

    # --- Days ---
    day_patterns = [
        r"^(\d+)\s*d$", r"^(\d+)\s*day$", r"^(\d+)\s*days$"
    ]
    for p in day_patterns:
        match = re.match(p, time_str)
        if match:
            return int(match.group(1)) * 24 * 60

    return None

def format_time(minutes: int):
    """
    Converts total minutes to human-friendly format, e.g.:
    8 -> 8 minutes
    125 -> 2 hours 5 minutes
    """
    hours = minutes // 60
    mins = minutes % 60
    parts = []
    if hours > 0:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if mins > 0:
        parts.append(f"{mins} minute" + ("s" if mins != 1 else ""))
    return " ".join(parts) if parts else "0 minutes"


# -------- SENTIMENT HELPER --------
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, label = torch.max(probs, dim=1)

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[label.item()], confidence.item()


# -------- COG --------
class Sentiment(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def analyze(self, ctx, *, time_input: str = "24h"):
        """
        Export messages from a custom time window.
        Accepts inputs like 5m, 2h, 1d, etc.
        """
        minutes = parse_time_input(time_input)
        if minutes is None:
            await ctx.send(
                "❌ Invalid time format.\n"
                "Examples:\n"
                "`5m`, `5mn`, `5 mins`, `5 minutes`\n"
                "`2h`, `2hr`, `2 hours`\n"
                "`1d`, `2 days`"
            )
            return

        # Minimum 5 minutes, maximum 7 days
        MIN_MINUTES = 5
        MAX_MINUTES = 7 * 24 * 60

        if minutes < MIN_MINUTES:
            await ctx.send("⚠️ Minimum time allowed is **5 minutes**.")
            return
        if minutes > MAX_MINUTES:
            await ctx.send("⚠️ Maximum time allowed is **7 days**.")
            return

        now = datetime.utcnow()
        since_time = now - timedelta(minutes=minutes)
        friendly_time = format_time(minutes)

        await ctx.send(f"🕒 Fetching messages from the last **{friendly_time}**...")

        # Fetch messages
        messages = []
        async for message in ctx.channel.history(limit=None, after=since_time):
            if message.author.bot or message.content.startswith("!"):
                continue
            messages.append([message.created_at.isoformat(), message.author.name, message.content])

        if not messages:
            await ctx.send("⚠️ No messages found in that time range.")
            return

        # Analyze messages
        results = []
        for msg in messages:
            content = msg[2].strip()
            if not content:
                continue
            sentiment, confidence = analyze_sentiment(content)
            results.append((msg[1], content, sentiment, confidence))

        # Summary
        total = len(results)
        pos = sum(1 for _, _, s, _ in results if s == "Positive")
        neu = sum(1 for _, _, s, _ in results if s == "Neutral")
        neg = sum(1 for _, _, s, _ in results if s == "Negative")
        pos_pct = pos / total * 100
        neu_pct = neu / total * 100
        neg_pct = neg / total * 100
        avg_conf = sum(c for _, _, _, c in results) / total * 100

        embed = discord.Embed(
            title=f"📊 Sentiment Summary (Last {friendly_time})",
            color=0x5865F2
        )
        embed.add_field(name="🟢 Positive", value=f"{pos} ({pos_pct:.1f}%)")
        embed.add_field(name="⚪ Neutral", value=f"{neu} ({neu_pct:.1f}%)")
        embed.add_field(name="🔴 Negative", value=f"{neg} ({neg_pct:.1f}%)")
        embed.add_field(name="📦 Total Messages", value=str(total), inline=False)
        embed.add_field(name="🎯 Avg Confidence", value=f"{avg_conf:.2f}%", inline=False)

        await ctx.send(embed=embed)

        # Save CSV
        csv_file = os.path.join(DATA_DIR, "analyzed_messages.csv")
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["author", "message", "sentiment", "confidence"])
            writer.writerows(results)

        await ctx.send("✅ Sentiment analysis complete. File saved.")


async def setup(bot):
    await bot.add_cog(Sentiment(bot))
