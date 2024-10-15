import tkinter as tk

from tkinter import *

from tkinter import messagebox

import requests

root = tk.Tk()
root.geometry('300x300')
root.maxsize(300,300)
root.minsize(300,300)
root.title('Send SMS')


def send_sms():
        
        number = phone_no.get()
        #To get Tkinter input from the text box
        
        messages = message.get("1.0","end-1c")
        #"1.0" - means that the input should be read from line one, character zero (ie: the very first character)
        #"end" part means to read until the end of the text box is reached and "-1c" to delete one character 

        url = "https://www.fast2sms.com/dev/bulk"
        #fast2sms url
        
        api = "Enter Your API here" #Go to fast2sms.com and signup to get the free Api
        querystring = {
                       "authorization":api,  #Our api
                       "sender_id":"FSTSMS", #Sender id to be shown in message
                       "message":messages,   #Message Inputted by the user
                       "language":"english", #language-English
                       "route":"p",          #Enter p for promotional and t for transactional
                       "numbers":number      #Number inputted by user
                       }

        headers = {
                 'cache-control': "no-cache"
                 #The no-cache directive means that a browser may cache a response,
                 #but must first submit a validation request to an origin server.
                 }
        
        requests.request("GET", url, headers=headers, params=querystring)
        #"GET" is used to request data from a specified resource
        #"url" The url of the request
        #"headers" A dictionary of HTTP headers to send to the specified url
        #"params" . A dictionary, list of tuples or bytes to send as a query string
        
        messagebox.showinfo("Send SMS",'SMS has been send successfully')
        #messagebox is a little popup showing a message to the user about

label = Label(root,text="Send SMS Using Python",font=('verdana',10,'bold'))
label.place(x=60,y=20)

phone_no = Entry(root,width=20,borderwidth=0,font=('verdana',10,'bold'))
phone_no.place(x=60,y=100)
phone_no.insert('end','phone number')

message = Text(root,height=5,width=25,borderwidth=0,font=('verdana',10,'bold'))
message.place(x=40,y=140)
message.insert('end','Message')

send = Button(root,text="Send Message",font=('verdana',10,'bold'),bg='blue',cursor='hand2',borderwidth=0,command=send_sms)
send.place(x=90,y=235)
root.mainloop()