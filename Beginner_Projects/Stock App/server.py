from flask import Flask, redirect, render_template, url_for, request
import requests

app = Flask(__name__)

API_key = '***************************'
api_url ="https://financialmodelingprep.com/api/v3/quote-short/{ticker}"

def fetch_price(ticker):
    data = requests.get(api_url.format(ticker=ticker.upper()), params = {'apikey': API_key}).json()
    for p in data:
        return p['price']
  

def fetch_volume(ticker):
    data = requests.get(api_url.format(ticker=ticker.upper()), params = {'apikey': API_key}).json()
    for p in data:
        return p['volume']



def fetch_income_stat(ticker):
    api_url = 'https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter?limit=8'
    financials = requests.get(api_url.format(ticker=ticker.upper()), params={'apikey':API_key}).json()
    financials.sort(key=lambda quarter: quarter['date'])
    return financials


def financial_chart(ticker):
    data =  fetch_income_stat(ticker)
    chart_data= [float(q['eps']) for q in data if q['eps']]
    chart_process= {"type": 'line',
                    'data':{
                        'labels':[q['date'] for q in data if q['eps']],
                        'datasets':[{'label':'EPS','data':chart_data}]
                    }
                }
    return chart_process


def profile(ticker):
    company_profile = requests.get(f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={API_key}").json()
    return company_profile



def todayQuote(ticker):
    quotes = requests.get(f"https://financialmodelingprep.com/api/v3/quote/{ticker.upper()}?apikey={API_key}").json()
    try:
        return quotes
    except Exception:
        return ("Quotes not avialble right now!")


@app.route("/stock/<ticker>")
def fetch_ticker(ticker):
    price = fetch_price(ticker)
    volume = fetch_volume(ticker)
    financials = fetch_income_stat(ticker)
    epsChart= financial_chart(ticker)
    companyprofile = profile(ticker)
    quotes = todayQuote(ticker)
    return render_template("stockpage.html", price=price, volume=volume, financials=financials, epsChart=epsChart, profile=companyprofile, quotes=quotes, ticker=ticker)



@app.route("/search", methods=['POST'])
def search():
    return redirect(url_for('fetch_ticker', ticker=request.form['searchticker']))




@app.route("/")
def home_page():
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True, threaded=True)