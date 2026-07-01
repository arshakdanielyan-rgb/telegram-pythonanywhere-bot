"""
Mocks all external dependencies before any bot module is imported.
This lets tests run without real API keys or network connections.
"""

import os
import sys
from unittest.mock import MagicMock

# ── Fake environment variables ─────────────────────────────────────────────────
os.environ["TELEGRAM_BOT_TOKEN"] = "1234567890:fake_token"
os.environ["AI_API_KEY"] = "fake_api_key"
# Prevent bot/config.py's webhook-secret bootstrap from creating a
# .webhook_secret file in the working tree at import time. Tests that
# need to exercise the bootstrap logic pass `file_path` explicitly.
os.environ["WEBHOOK_SECRET"] = "fake_webhook_secret_for_tests"

# ── Mock external packages ─────────────────────────────────────────────────────
mock_bot_instance = MagicMock()
mock_bot_instance.get_me.return_value = MagicMock(id=42, username="testbot")
# Decorators must pass through so handler functions remain callable
mock_bot_instance.message_handler.return_value = lambda f: f
mock_bot_instance.callback_query_handler.return_value = lambda f: f

mock_telebot = MagicMock()
mock_telebot.TeleBot.return_value = mock_bot_instance

# Flask mock: make @app.route() pass through too
mock_flask = MagicMock()
mock_flask_app = MagicMock()
mock_flask_app.route.return_value = lambda f: f
mock_flask.Flask.return_value = mock_flask_app

sys.modules["telebot"] = mock_telebot
sys.modules["telebot.types"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["flask"] = mock_flask
sys.modules["gradio_client"] = MagicMock()
