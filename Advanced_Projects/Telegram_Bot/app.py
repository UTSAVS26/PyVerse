import logging
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
WEATHER_API_KEY = 'your_weather_api_key'
TELEGRAM_TOKEN = 'your_telegram_bot_token'  

# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Welcome! Use /help to see available commands.")

# Help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = "/start - Start the bot\n/help - Get help\n/weather - Get current weather\n/motivate - Get daily motivation"
    await update.message.reply_text(help_text)

# Weather command with API integration
async def weather(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    city = ' '.join(context.args) if context.args else 'London'
    response = requests.get(f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}")
    if response.status_code == 200:
        data = response.json()
        weather_info = f"Weather in {data['location']['name']}: {data['current']['temp_c']}°C, {data['current']['condition']['text']}"
        await update.message.reply_text(weather_info)
    else:
        await update.message.reply_text("Could not retrieve weather data.")

# Motivation command with inline keyboard
async def motivate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    quote = "Stay positive, work hard, make it happen!"
    keyboard = [[InlineKeyboardButton("Get Another", callback_data='new_motivation')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(quote, reply_markup=reply_markup)

# Callback for inline button to get new quote
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    new_quote = "Believe in yourself and all that you are!"
    await query.edit_message_text(text=new_quote)

# Main function to start the bot
def main() -> None:
    # Create the Application instance
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("weather", weather))
    application.add_handler(CommandHandler("motivate", motivate))
    application.add_handler(CallbackQueryHandler(button))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()
