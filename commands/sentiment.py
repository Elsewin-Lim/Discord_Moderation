import discord
from discord.ext import commands
from datetime import datetime, timedelta
import csv
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = os.path.abspath(r"C:\Users\U-ser\OneDrive - American University of Phnom Penh\Desktop\AUPP\Natural Language Processing\Final Project\Discord Bot\models\sentimental")
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load sentiment model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, label = torch.max(probs, dim=1)

    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
    return label_map[label.item()], confidence.item()


class Sentiment(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def export(self, ctx):
        """Export messages from last 24 hours and analyze sentiment."""
        channel = ctx.channel
        now = datetime.utcnow()
        one_day_ago = now - timedelta(days=1)

        await ctx.send("🕒 Fetching 24h messages...")

        # Fetch messages
        messages = []
        async for message in channel.history(limit=None, after=one_day_ago):
            if message.author.bot:
                continue
            if message.content.startswith("!"):
                continue
            messages.append([message.created_at.isoformat(), message.author.name, message.content])

        if not messages:
            await ctx.send("⚠️ No messages found.")
            return

        # Analyze
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

        # Embed summary
        embed = discord.Embed(
            title="📊 Sentiment Analysis Summary (Last 24 Hours)",
            color=0x5865F2
        )
        embed.add_field(name="🟢 Positive", value=f"{pos} ({pos_pct:.1f}%)", inline=True)
        embed.add_field(name="⚪ Neutral", value=f"{neu} ({neu_pct:.1f}%)", inline=True)
        embed.add_field(name="🔴 Negative", value=f"{neg} ({neg_pct:.1f}%)", inline=True)
        embed.add_field(name="📦 Total", value=str(total), inline=False)
        embed.add_field(name="🎯 Avg Confidence", value=f"{avg_conf:.2f}%", inline=False)

        await ctx.send(embed=embed)

        # Save CSV
        csv_file = os.path.join(DATA_DIR, "analyzed_messages.csv")
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["author", "message", "sentiment", "confidence"])
            writer.writerows(results)

        await ctx.send("✅ Sentiment analysis complete.")

async def setup(bot):
    await bot.add_cog(Sentiment(bot))
