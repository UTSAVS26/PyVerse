import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Flask settings
    DEBUG = os.environ.get('FLASK_DEBUG') or True
    
    # Data paths
    DATA_PATH = os.path.join(basedir, 'data/bitcoin_dataset.csv')
    
    # Model settings
    DEFAULT_TEST_SIZE = 0.2
    RANDOM_STATE = 42