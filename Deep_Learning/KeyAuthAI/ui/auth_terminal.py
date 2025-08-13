"""
Authentication Terminal Interface for KeyAuthAI

This module provides a command-line interface for user registration and authentication:
- User registration with keystroke data collection
- User authentication with real-time verification
- Interactive session management
- Color-coded output for better UX
"""

import argparse
import sys
import os
from typing import Dict, List, Optional, Any
from colorama import init, Fore, Back, Style

# Initialize colorama for cross-platform colored output
init()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.keystroke_logger import KeystrokeLogger
from model.train_model import KeystrokeModelTrainer
from model.verify_user import UserVerifier


class AuthTerminal:
    """Command-line interface for KeyAuthAI authentication."""
    
    def __init__(self):
        """Initialize the authentication terminal."""
        self.logger = KeystrokeLogger()
        self.trainer = KeystrokeModelTrainer()
        self.verifier = UserVerifier()
        
        # Default passphrase for training
        self.default_passphrase = "the quick brown fox jumps over the lazy dog"
    
    def register_user(self, username: str, passphrase: str = None, 
                     sessions: int = 5, model_type: str = 'svm') -> Dict[str, Any]:
        """
        Register a new user with keystroke dynamics.
        
        Args:
            username: Name of the user
            passphrase: Passphrase to type (optional)
            sessions: Number of training sessions
            model_type: Type of model to train
            
        Returns:
            Dictionary with registration results
        """
        if passphrase is None:
            passphrase = self.default_passphrase
        
        print(f"{Fore.CYAN}🕵️ KeyAuthAI User Registration{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}User: {username}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Passphrase: '{passphrase}'{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Training sessions: {sessions}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Model type: {model_type}{Style.RESET_ALL}")
        print()
        
        # Check if user already exists
        if username in self.logger.list_users():
            print(f"{Fore.RED}⚠️  User '{username}' already exists!{Style.RESET_ALL}")
            response = input("Do you want to overwrite existing data? (y/N): ").lower()
            if response != 'y':
                return {'success': False, 'error': 'Registration cancelled by user'}
            
            # Delete existing user data
            self.logger.delete_user_data(username)
        
        # Collect training sessions
        collected_sessions = []
        
        for session_num in range(1, sessions + 1):
            print(f"\n{Fore.GREEN}📝 Training Session {session_num}/{sessions}{Style.RESET_ALL}")
            print(f"Please type: '{passphrase}'")
            print("Press Enter when you've finished typing...")
            print(f"{Fore.BLUE}Tip: Try to type naturally and consistently{Style.RESET_ALL}")
            
            try:
                # Record keystroke session
                self.logger.start_recording(username, passphrase)
                input()
                session_data = self.logger.stop_recording()
                
                if len(session_data) > 0:
                    collected_sessions.append(session_data)
                    print(f"{Fore.GREEN}✅ Session {session_num} recorded successfully{Style.RESET_ALL}")
                    print(f"   Keystroke events: {len(session_data)}")
                else:
                    print(f"{Fore.RED}❌ Session {session_num} failed - no keystrokes detected{Style.RESET_ALL}")
                    
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}⚠️  Registration cancelled by user{Style.RESET_ALL}")
                return {'success': False, 'error': 'Registration cancelled by user'}
            except Exception as e:
                print(f"{Fore.RED}❌ Error recording session {session_num}: {e}{Style.RESET_ALL}")
        
        if len(collected_sessions) < 3:
            print(f"{Fore.RED}❌ Not enough valid sessions collected. Need at least 3.{Style.RESET_ALL}")
            return {'success': False, 'error': 'Insufficient training data'}
        
        # Train model
        print(f"\n{Fore.CYAN}🤖 Training {model_type} model...{Style.RESET_ALL}")
        
        try:
            training_results = self.trainer.train_model(username, model_type, min_sessions=3)
            model_path = self.trainer.save_model(username)
            
            print(f"{Fore.GREEN}✅ Model trained successfully!{Style.RESET_ALL}")
            print(f"   Model type: {training_results['model_type']}")
            print(f"   Sessions used: {training_results['n_sessions']}")
            print(f"   Features extracted: {training_results['n_features']}")
            
            if 'accuracy' in training_results:
                print(f"   Accuracy: {training_results['accuracy']:.4f}")
            if 'anomaly_rate' in training_results:
                print(f"   Anomaly rate: {training_results['anomaly_rate']:.4f}")
            
            return {
                'success': True,
                'username': username,
                'sessions_collected': len(collected_sessions),
                'model_path': model_path,
                'training_results': training_results
            }
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error training model: {e}{Style.RESET_ALL}")
            return {'success': False, 'error': f'Model training failed: {e}'}
    
    def authenticate_user(self, username: str, passphrase: str = None) -> Dict[str, Any]:
        """
        Authenticate a user using keystroke dynamics.
        
        Args:
            username: Name of the user
            passphrase: Passphrase to type (optional)
            
        Returns:
            Dictionary with authentication results
        """
        print(f"{Fore.CYAN}🔐 KeyAuthAI User Authentication{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}User: {username}{Style.RESET_ALL}")
        
        # Check if user exists
        if username not in self.logger.list_users():
            print(f"{Fore.RED}❌ User '{username}' not found!{Style.RESET_ALL}")
            return {'success': False, 'error': 'User not found'}
        
        # Get user stats
        stats = self.verifier.get_user_stats(username)
        if 'error' in stats:
            print(f"{Fore.RED}❌ Error getting user stats: {stats['error']}{Style.RESET_ALL}")
            return {'success': False, 'error': stats['error']}
        
        print(f"   Sessions: {stats['n_sessions']}")
        print(f"   Available models: {', '.join(stats['available_models'])}")
        print()
        
        # Get passphrase
        if passphrase is None:
            passphrase = stats['passphrase']
        
        print(f"Please type: '{passphrase}'")
        print("Press Enter when you've finished typing...")
        
        try:
            # Verify user
            result = self.verifier.verify_user_interactive(username, passphrase)
            
            print(f"\n{Fore.CYAN}🔍 Authentication Results{Style.RESET_ALL}")
            print("=" * 40)
            
            if result['authenticated']:
                print(f"{Fore.GREEN}✅ Authentication SUCCESSFUL!{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ Authentication FAILED!{Style.RESET_ALL}")
            
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Threshold: {result['threshold']:.4f}")
            print(f"Model Type: {result.get('model_type', 'N/A')}")
            print(f"Session Length: {result.get('session_length', 'N/A')}")
            
            if result['error']:
                print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            
            return {
                'success': True,
                'authenticated': result['authenticated'],
                'confidence': result['confidence'],
                'model_type': result.get('model_type'),
                'session_length': result.get('session_length'),
                'error': result.get('error')
            }
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠️  Authentication cancelled by user{Style.RESET_ALL}")
            return {'success': False, 'error': 'Authentication cancelled by user'}
        except Exception as e:
            print(f"{Fore.RED}❌ Authentication error: {e}{Style.RESET_ALL}")
            return {'success': False, 'error': f'Authentication error: {e}'}
    
    def list_users(self) -> List[str]:
        """List all registered users."""
        users = self.logger.list_users()
        
        if not users:
            print(f"{Fore.YELLOW}📋 No users registered yet.{Style.RESET_ALL}")
            return []
        
        print(f"{Fore.CYAN}📋 Registered Users{Style.RESET_ALL}")
        print("=" * 30)
        
        for i, username in enumerate(users, 1):
            stats = self.verifier.get_user_stats(username)
            if 'error' not in stats:
                print(f"{i}. {Fore.GREEN}{username}{Style.RESET_ALL}")
                print(f"   Sessions: {stats['n_sessions']}")
                print(f"   Models: {', '.join(stats['available_models'])}")
            else:
                print(f"{i}. {Fore.RED}{username} (error){Style.RESET_ALL}")
        
        return users
    
    def delete_user(self, username: str) -> bool:
        """
        Delete a user and all associated data.
        
        Args:
            username: Name of the user to delete
            
        Returns:
            True if successful, False otherwise
        """
        if username not in self.logger.list_users():
            print(f"{Fore.RED}❌ User '{username}' not found!{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.YELLOW}⚠️  Are you sure you want to delete user '{username}'?{Style.RESET_ALL}")
        print("This will delete all keystroke data and trained models.")
        response = input("Type 'DELETE' to confirm: ")
        
        if response != 'DELETE':
            print(f"{Fore.YELLOW}⚠️  Deletion cancelled.{Style.RESET_ALL}")
            return False
        
        try:
            # Delete user data
            self.logger.delete_user_data(username)
            
            # Delete model files
            model_types = ['svm', 'random_forest', 'knn', 'one_class_svm', 'isolation_forest']
            for model_type in model_types:
                model_path = f"model/model_{username}_{model_type}.pkl"
                if os.path.exists(model_path):
                    os.remove(model_path)
            
            print(f"{Fore.GREEN}✅ User '{username}' deleted successfully!{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error deleting user: {e}{Style.RESET_ALL}")
            return False
    
    def interactive_mode(self):
        """Run the terminal in interactive mode."""
        print(f"{Fore.CYAN}🕵️ KeyAuthAI Interactive Terminal{Style.RESET_ALL}")
        print("=" * 40)
        
        while True:
            print(f"\n{Fore.BLUE}Available commands:{Style.RESET_ALL}")
            print("1. register <username> - Register a new user")
            print("2. login <username> - Authenticate a user")
            print("3. list - List all users")
            print("4. delete <username> - Delete a user")
            print("5. quit - Exit the program")
            
            try:
                command = input(f"\n{Fore.GREEN}KeyAuthAI> {Style.RESET_ALL}").strip().split()
                
                if not command:
                    continue
                
                if command[0] == 'quit':
                    print(f"{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                    break
                
                elif command[0] == 'register':
                    if len(command) < 2:
                        print(f"{Fore.RED}❌ Usage: register <username>{Style.RESET_ALL}")
                        continue
                    
                    username = command[1]
                    result = self.register_user(username)
                    
                    if result['success']:
                        print(f"{Fore.GREEN}✅ Registration successful!{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}❌ Registration failed: {result['error']}{Style.RESET_ALL}")
                
                elif command[0] == 'login':
                    if len(command) < 2:
                        print(f"{Fore.RED}❌ Usage: login <username>{Style.RESET_ALL}")
                        continue
                    
                    username = command[1]
                    result = self.authenticate_user(username)
                    
                    if result['success'] and result['authenticated']:
                        print(f"{Fore.GREEN}✅ Login successful!{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}❌ Login failed!{Style.RESET_ALL}")
                
                elif command[0] == 'list':
                    self.list_users()
                
                elif command[0] == 'delete':
                    if len(command) < 2:
                        print(f"{Fore.RED}❌ Usage: delete <username>{Style.RESET_ALL}")
                        continue
                    
                    username = command[1]
                    self.delete_user(username)
                
                else:
                    print(f"{Fore.RED}❌ Unknown command: {command[0]}{Style.RESET_ALL}")
            
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='KeyAuthAI Authentication Terminal')
    parser.add_argument('--register', help='Register a new user')
    parser.add_argument('--login', help='Authenticate a user')
    parser.add_argument('--list', action='store_true', help='List all users')
    parser.add_argument('--delete', help='Delete a user')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--passphrase', help='Custom passphrase for registration/login')
    parser.add_argument('--sessions', type=int, default=5, help='Number of training sessions')
    parser.add_argument('--model', default='svm', 
                       choices=['svm', 'random_forest', 'knn', 'one_class_svm', 'isolation_forest'],
                       help='Model type for training')
    
    args = parser.parse_args()
    
    terminal = AuthTerminal()
    
    try:
        if args.interactive:
            terminal.interactive_mode()
        elif args.register:
            result = terminal.register_user(args.register, args.passphrase, args.sessions, args.model)
            if result['success']:
                print(f"{Fore.GREEN}✅ Registration successful!{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ Registration failed: {result['error']}{Style.RESET_ALL}")
                sys.exit(1)
        elif args.login:
            result = terminal.authenticate_user(args.login, args.passphrase)
            if result['success'] and result['authenticated']:
                print(f"{Fore.GREEN}✅ Authentication successful!{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ Authentication failed!{Style.RESET_ALL}")
                sys.exit(1)
        elif args.list:
            terminal.list_users()
        elif args.delete:
            success = terminal.delete_user(args.delete)
            if not success:
                sys.exit(1)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main() 