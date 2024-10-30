from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import argparse
import logging
import sys
import warnings

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RickMortyBot:
    def __init__(self, model_path='output-small', device=None):
        """
        Initialize the RickMorty chatbot.
        
        Args:
            model_path (str): Path to the fine-tuned model
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
            
            logger.info("Loading model...")
            self.model = AutoModelWithLMHead.from_pretrained(model_path)
            self.model.to(self.device)
            
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

        self.chat_history_ids = None
        
    def generate_response(self, user_input, max_length=200):
        """
        Generate a response to the user input.
        
        Args:
            user_input (str): The user's input text
            max_length (int): Maximum length of the generated response
            
        Returns:
            str: The model's response
        """
        try:
            # Encode the user input
            new_user_input_ids = self.tokenizer.encode(
                user_input + self.tokenizer.eos_token,
                return_tensors='pt'
            ).to(self.device)

            # Append to chat history if it exists
            bot_input_ids = (
                torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
                if self.chat_history_ids is not None
                else new_user_input_ids
            )

            # Generate response
            self.chat_history_ids = self.model.generate(
                bot_input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8
            )

            # Decode and return the response
            response = self.tokenizer.decode(
                self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )
            
            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Wubba lubba dub dub! (Error generating response)"
    
    def reset_chat_history(self):
        """Reset the chat history"""
        self.chat_history_ids = None
        logger.info("Chat history reset")

def interactive_chat(bot, max_turns=5):
    """
    Start an interactive chat session with the bot.
    
    Args:
        bot (RickMortyBot): The initialized chatbot
        max_turns (int): Maximum number of conversation turns
    """
    print("\nWelcome to the Rick and Morty ChatBot!")
    print("Type 'quit' to exit, 'reset' to start a new conversation")
    print("-" * 50)
    
    turn = 0
    while turn < max_turns:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
            
        if user_input.lower() == 'reset':
            bot.reset_chat_history()
            print("\nStarting new conversation...")
            continue
            
        if not user_input:
            print("Please say something!")
            continue
            
        response = bot.generate_response(user_input)
        print(f"Bot: {response}")
        
        turn += 1
        
    if turn >= max_turns:
        print("\nMaximum conversation length reached. Starting new conversation...")
        bot.reset_chat_history()

def main():
    parser = argparse.ArgumentParser(description='Test the Rick and Morty chatbot')
    parser.add_argument('--model_path', type=str, default='output-small',
                        help='Path to the fine-tuned model')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], 
                        help='Device to run the model on')
    parser.add_argument('--max_turns', type=int, default=10,
                        help='Maximum number of conversation turns')
    
    args = parser.parse_args()
    
    try:
        bot = RickMortyBot(model_path=args.model_path, device=args.device)
        interactive_chat(bot, max_turns=args.max_turns)
    except KeyboardInterrupt:
        print("\nChat session ended by user.")
    except Exception as e:
        logger.error(f"Error during chat session: {str(e)}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()