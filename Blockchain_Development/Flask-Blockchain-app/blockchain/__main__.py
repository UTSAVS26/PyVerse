from configparser import ConfigParser

from .app import app

# Get the config for the Flask app
config = ConfigParser()
config.read("config.ini")


if __name__ == "__main__":
    app.run(
        host=config.get("HOST", "host"),
        port=config.getint("HOST", "port"),
        debug=config.getboolean("DEBUG", "debug"),
    )
