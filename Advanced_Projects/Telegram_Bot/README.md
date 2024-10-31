# Advanced Telegram Bot Project

This project is a feature-rich Telegram bot built using the `python-telegram-bot` library. The bot provides a variety of interactive commands and functionalities, such as weather updates, motivational quotes, and more. 

## Features

- **Basic Commands**: Includes `/start`, `/help`, and `/info` commands to guide users.
- **Weather Updates**: Provides real-time weather information for any city, using an external API.
- **Motivational Quotes**: Offers daily motivational quotes with an option to get more via inline buttons.
- **Interactive Elements**: Inline keyboards for quick interactions, making the bot more engaging.
- **Error Handling**: Comprehensive logging and error handling for smooth operation.

## Setup and Installation

### 1. Install Dependencies

Ensure you have Python 3 installed. Then, install the required libraries:

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

- **Telegram Bot API Key**: Create a bot through [BotFather on Telegram](https://core.telegram.org/bots#botfather) and get the token.
- **Weather API Key**: Sign up on [WeatherAPI](https://www.weatherapi.com/) or [OpenWeatherMap](https://openweathermap.org/) to obtain an API key.

### 3. Configure API Keys

Replace placeholders in `app.py` with your actual API keys:

```python
WEATHER_API_KEY = 'your_weather_api_key'
TELEGRAM_TOKEN = 'your_telegram_bot_token'
```

### 4. Run the Bot

Run the bot using:

```bash
python app.py
```

## Usage

Once the bot is running, you can interact with it on Telegram:

- **/start** - Starts the bot and displays a welcome message.
- **/help** - Lists available commands.
- **/weather `<city_name>`** - Retrieves the current weather for a specified city.
- **/motivate** - Sends a motivational quote with an inline option to get more quotes.

## Example Interactions

- `/start`: "Welcome! Use /help to see available commands."
- `/weather London`: "Weather in London: 18°C, Partly Cloudy"
- `/motivate`: "Stay positive, work hard, make it happen!" with a button for more quotes.

## Project Structure

```
├── app.py                 # Main bot code
├── README.md              # Project documentation
└── requirements.txt       # Dependencies
```

## Troubleshooting

- **TypeError**: If you encounter issues with the `Updater` class, ensure you're using `Application` instead (for `python-telegram-bot` v20+).
- **Network/API Errors**: Ensure API keys are correctly configured and active.

