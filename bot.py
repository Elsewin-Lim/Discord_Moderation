import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import asyncio

# ---- LOAD TOKEN ----
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise ValueError("⚠️ DISCORD_TOKEN not found in .env file!")

# ---- DISCORD SETUP ----
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ---- EXTENSIONS ----
extensions = [
    "commands.sentiment",
    "commands.topic_filter"
]

async def load_extensions():
    for ext in extensions:
        try:
            await bot.load_extension(ext)
            print(f"✅ Loaded extension: {ext}")
        except Exception as e:
            print(f"❌ Failed to load {ext}: {e}")

# ---- EVENTS ----
@bot.event
async def on_ready():
    print(f"🤖 Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")

# ---- MAIN ----
async def main():
    await load_extensions()
    await bot.start(TOKEN)

# Run the bot
asyncio.run(main())
