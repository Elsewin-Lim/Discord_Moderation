# 🤖 Discord Moderation System

NLP-powered Discord bot that analyzes message sentiment and optionally filters messages by topic.
Uses local Hugging Face RoBERTa models (provided in `./models/`) and stores logs in `./data/`.

---

## ⚙️ Features

* Real-time sentiment classification (Positive / Negative / Neutral)
* Real-time topic filtering (World, Sports, Business, Sci/Tech)
* One-command analysis of the last 24 hours of messages in a channel
* Clean summary embeds with counts, percentages, and average model confidence
* Local CSV logging of raw and analyzed messages
* Automatic message deletion for off-topic messages after warning

---

## Discord Bot Link Invitation

```
https://discord.com/oauth2/authorize?client_id=1434914563421110312&permissions=17179978784&integration_type=0&scope=bot
```

---

## 🔧 Requirements

* Python 3.9+
* Model files placed in `./models/` (`config.json`, `model.safetensors`, `vocab.json`, etc.)
* Bot token with appropriate permissions

Install Python packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the bot

```bash
python bot.py
```

---

## 💬 Commands & Usage

### Sentiment Analysis Commands

* `!export`
  Fetches messages from the last 24 hours in the current channel, runs sentiment analysis, and posts a summary embed with counts, percentages, and average model confidence.


### Topic Filter Commands

* `!topicset <topic>`
  Set the allowed topic for the current channel.
  Example:

  ```
  !topicset Sports
  ```

* `!topicget`
  Check the current topic set for this channel.
  Example:

  ```
  !topicget
  ```

* `!topiclist`
  Show all available topics that the bot can classify.
  Example:

  ```
  !topiclist
  ```

* `!topicclear`
  Removes topic filtering from the current channel.
  Example:

  ```
  !topicclear
  ```

  The bot will monitor messages in that channel and warn or delete messages that do not match the set topic.

**Allowed Topics:**

```
World, Sports, Business, Sci/Tech
```

* How it works:

  * The bot predicts the topic of every non-command message.
  * If the message topic does not match the allowed topic, the bot:

    1. Sends a warning reply to the user.
    2. Logs the message and predicted topic in `./data/filtered_messages.csv`.
    3. Deletes the message after 10 seconds (if not corrected).

---

### Notes

* The bot only works with the pre-defined topics above.
* Sentiment analysis and topic filtering run **locally** using your Hugging Face RoBERTa models.
* Make sure your model directories contain **all required files** (`config.json`, `model.safetensors`, tokenizer files, etc.)

---

