import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
from steg.image_steg import encode_message, decode_message, get_max_message_length
from crypto.aes import generate_key, encrypt_message, decrypt_message, save_key_info, load_key_info

class StegaChatUI:
    def __init__(self, root):
        self.root = root
        self.root.title("StegaChat - Secure Image Steganography")
        self.root.geometry("600x500")
        
        # Store current encryption key and salt
        self.current_fernet = None
        self.current_salt = None
        
        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="StegaChat", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Message input section
        input_frame = tk.LabelFrame(main_frame, text="Message Input", padx=10, pady=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(input_frame, text="Enter your message:").pack(anchor=tk.W)
        self.text_entry = tk.Text(input_frame, height=4, width=50)
        self.text_entry.pack(fill=tk.X, pady=(5, 10))
        
        # Encryption section
        encrypt_frame = tk.LabelFrame(main_frame, text="Encryption", padx=10, pady=10)
        encrypt_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.encrypt_var = tk.IntVar()
        self.encrypt_check = tk.Checkbutton(encrypt_frame, text="Encrypt message", variable=self.encrypt_var, 
                                          command=self.on_encrypt_toggle)
        self.encrypt_check.pack(anchor=tk.W)
        
        self.password_frame = tk.Frame(encrypt_frame)
        self.password_entry = tk.Entry(self.password_frame, show="*", width=30)
        self.password_entry.pack(side=tk.LEFT, padx=(20, 5))
        tk.Label(self.password_frame, text="Password:").pack(side=tk.LEFT)
        
        # Buttons section
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.encode_button = tk.Button(button_frame, text="Encode & Save", command=self.encode_message, 
                                     bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.encode_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.decode_button = tk.Button(button_frame, text="Decode Message", command=self.decode_message,
                                     bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        self.decode_button.pack(side=tk.LEFT)
        
        # Status section
        status_frame = tk.LabelFrame(main_frame, text="Status", padx=10, pady=10)
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_text = tk.Text(status_frame, height=8, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Initially hide password frame
        self.password_frame.pack_forget()

    def on_encrypt_toggle(self):
        """Show/hide password input based on encryption checkbox"""
        if self.encrypt_var.get():
            self.password_frame.pack(anchor=tk.W, pady=(5, 0))
        else:
            self.password_frame.pack_forget()
            self.current_fernet = None
            self.current_salt = None

    def log_status(self, message: str):
        """Add message to status log"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def encode_message(self):
        """Encode message into an image"""
        try:
            # Get message
            message = self.text_entry.get("1.0", tk.END).strip()
            if not message:
                messagebox.showerror("Error", "Please enter a message to encode.")
                return
            
            # Get input image
            img_path = filedialog.askopenfilename(
                title="Select input image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
            )
            if not img_path:
                return
            
            # Check if image is large enough
            try:
                max_length = get_max_message_length(img_path)
                if len(message) > max_length:
                    messagebox.showerror("Error", 
                                       f"Message too long. Maximum {max_length} characters allowed for this image.")
                    return
            except Exception as e:
                messagebox.showerror("Error", f"Could not analyze image: {e}")
                return
            
            # Handle encryption
            if self.encrypt_var.get():
                password = self.password_entry.get()
                if not password:
                    messagebox.showerror("Error", "Please enter a password for encryption.")
                    return
                
                try:
                    self.current_fernet, self.current_salt = generate_key(password)
                    encrypted_message = encrypt_message(self.current_fernet, message)
                    import base64
                    message = base64.b64encode(encrypted_message).decode('ascii')
                    self.log_status("Message encrypted successfully")
                except Exception as e:
                    messagebox.showerror("Error", f"Encryption failed: {e}")
                    return
            
            # Get output path
            output_path = filedialog.asksaveasfilename(
                title="Save encoded image as",
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
            )
            if not output_path:
                return
            
            # Encode the message
            success = encode_message(img_path, message, output_path)
            if success:
                # Save key info if encrypted
                if self.encrypt_var.get() and self.current_salt:
                    key_info_path = output_path + ".key"
                    save_key_info(self.current_fernet, self.current_salt, key_info_path)
                    self.log_status(f"Key info saved to: {key_info_path}")
                
                self.log_status(f"Message encoded successfully: {output_path}")
                messagebox.showinfo("Success", f"Message encoded and saved to:\n{output_path}")
            else:
                messagebox.showerror("Error", "Failed to encode message.")
                
        except Exception as e:
            self.log_status(f"Error: {e}")
            messagebox.showerror("Error", f"Encoding failed: {e}")

    def decode_message(self):
        """Decode message from an image"""
        try:
            # Get input image
            img_path = filedialog.askopenfilename(
                title="Select image to decode",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
            )
            if not img_path:
                return
            
            # Decode the message
            message = decode_message(img_path)
            
            # Check if message is encrypted
            # Check if message is encrypted (assuming base64 encoding)
            try:
                import base64
                decoded = base64.b64decode(message)
                if decoded.startswith(b'gAAAAA'):  # Fernet encrypted messages start with this
                    is_encrypted = True
                else:
                    is_encrypted = False
            except Exception:
                is_encrypted = False

            if is_encrypted:
                # Ask for password
                password = simpledialog.askstring("Password Required", 
                                                "This message appears to be encrypted. Enter the password:")
                if not password:
                    return
                
                try:
                    # Try to load key info
                    key_info_path = img_path + ".key"
                    if os.path.exists(key_info_path):
                        salt = load_key_info(key_info_path)
                        fernet, _ = generate_key(password, salt)
                    else:
                        # Fallback: generate new key (may not work if salt was different)
                        fernet, _ = generate_key(password)
                    
                    # Decrypt the message
                    # Decrypt the message (assuming base64 encoding)
                    import base64
                    encrypted_bytes = base64.b64decode(message)
                    decrypted_message = decrypt_message(fernet, encrypted_bytes)
                    message = decrypted_message
                    self.log_status("Message decrypted successfully")
                except Exception as e:
                    messagebox.showerror("Error", f"Decryption failed: {e}")
                    return
            
            # Display the message
            self.log_status(f"Decoded message: {message}")
            messagebox.showinfo("Decoded Message", message)
            
        except Exception as e:
            self.log_status(f"Error: {e}")
            messagebox.showerror("Error", f"Decoding failed: {e}")
