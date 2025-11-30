import discord
from discord.ext import commands, tasks
import os
import csv
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from datetime import datetime, timedelta

# ---- DATA DIR ----
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
FILTERED_FILE = os.path.join(DATA_DIR, "filtered_messages.csv")

# ---- MODEL PATH ----
MODEL_DIR = os.path.abspath(r"C:/Users/U-Ser/OneDrive - American University of Phnom Penh/Desktop/AUPP/Natural Language Processing/Final Project/Discord Bot/models/topic_classifier")

# ---- LOAD TOKENIZER & MODEL ----
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

# ---- LABEL MAP ----
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# ---- COG ----
class TopicFilter(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.channel_topics = {}  # store topic per channel

    # ---- SET TOPIC ----
    @commands.command(name="topicset")
    async def set_topic(self, ctx, *, topic: str):
        """Set the allowed topic for this channel."""
        topic = topic.strip()
        if topic.lower() not in [t.lower() for t in label_map.values()]:
            await ctx.send(f"❌ Invalid topic. Choose from: {', '.join(label_map.values())}")
            return
        self.channel_topics[ctx.channel.id] = topic
        await ctx.send(f"✅ Topic for this channel set to **{topic}**.")

    # ---- GET CURRENT TOPIC ----
    @commands.command(name="topicget")
    async def get_topic(self, ctx):
        """Get the current topic for this channel."""
        topic = self.channel_topics.get(ctx.channel.id, None)
        if topic:
            await ctx.send(f"ℹ️ Current topic for this channel is **{topic}**.")
        else:
            await ctx.send("ℹ️ No topic is set for this channel yet. Use `!topicset <topic>` to set one.")

    # ---- LIST AVAILABLE TOPICS ----
    @commands.command(name="topiclist")
    async def list_topics(self, ctx):
        """List all available topics the bot can classify."""
        await ctx.send(f"📌 Available topics: {', '.join(label_map.values())}")

    # ---- CLEAR TOPIC ----
    @commands.command(name="topicclear")
    async def clear_topic(self, ctx):
        """Clear/remove topic filtering for this channel."""
        if ctx.channel.id in self.channel_topics:
            del self.channel_topics[ctx.channel.id]
            await ctx.send("🧹 Topic filter cleared for this channel. All messages are now allowed.")
        else:
            await ctx.send("ℹ️ No topic is currently set for this channel.")

    # ---- CHECK MESSAGE TOPIC ----
    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot:
            return
        if message.content.startswith("!"):
            return

        channel_id = message.channel.id
        if channel_id not in self.channel_topics:
            return

        allowed_topic = self.channel_topics[channel_id]

        # ---- PREDICT TOPIC ----
        inputs = tokenizer(message.content, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            predicted_topic = label_map[predicted_class]

        if predicted_topic != allowed_topic:
            warning_msg = await message.reply(
                f"⚠️ This message seems off-topic (Predicted: {predicted_topic}). It will be deleted in 10 seconds."
            )

            # ---- SAVE TO CSV ----
            with open(FILTERED_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    message.author.name,
                    message.content,
                    predicted_topic,
                    allowed_topic
                ])

            # ---- COUNTDOWN & DELETE ----
            for i in range(10, 0, -1):
                await warning_msg.edit(content=f"⚠️ Off-topic message (Predicted: {predicted_topic}). Deleting in {i} seconds...")
                await discord.utils.sleep_until(datetime.utcnow() + timedelta(seconds=1))

            try:
                await message.delete()
                await warning_msg.edit(content=f"✅ Message deleted (Predicted: {predicted_topic})")
            except discord.NotFound:
                pass

# ---- SETUP ----
async def setup(bot):
    await bot.add_cog(TopicFilter(bot))
