from flask import Flask,render_template,url_for
import requests #request library 
from flask import request as req #request of flask

app=Flask(__name__)
@app.route("/",methods=["GET","POST"])
def Index():
    return render_template("index.html")

@app.route("/Summarize",methods=["GET","POST"])
def Summarize():
    if req.method=="POST":
        API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
        headers = {"Authorization": f"Bearer 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"}

        data=req.form["data"]
        maxL=int(req.form["MaxL"])
        minL=maxL//4
        
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
            
        output = query({
            "inputs": data,
            "parameters":{"min_length":minL,"max_length":maxL},
        })[0]  #these all are stored in key value pairs i.e. query is dictionary
        #print(output) 
        return render_template("index.html",result=output["summary_text"],mini=minL) #to send from backend to frontend
    else:
        return render_template("index.html")





if __name__=='__main__': #to execute code w/o writing in terminal
    app.debug=True       #true during testing, false during production
    app.run()