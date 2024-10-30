import requests
from django.shortcuts import render
from .forms import CurrencyConverterForm
from decimal import Decimal, ROUND_DOWN

API_KEY = "8db4f338e84f0e6b160b78ea"  # add your API key from exhangerate api


def currency_converter(request):
    result = None
    error_message = None
    if request.method == "POST":
        form = CurrencyConverterForm(request.POST)
        if form.is_valid():
            amount = Decimal(form.cleaned_data["amount"])  
            from_currency = form.cleaned_data["from_currency"].upper()
            to_currency = form.cleaned_data["to_currency"].upper()

            
            api_url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}?apikey={API_KEY}"

            try:
                
                response = requests.get(api_url)
                response.raise_for_status()  

              
                data = response.json()
                if to_currency in data["rates"]:
                    
                    exchange_rate = Decimal(data["rates"][to_currency])
                    result = amount * exchange_rate  # Perform conversion

                    
                    result = result.quantize(
                        Decimal("0.001"), rounding=ROUND_DOWN
                    ) 
                else:
                    error_message = "Invalid target currency."
            except requests.exceptions.RequestException as e:
                error_message = f"Error fetching data from the API: {e}"

    else:
        form = CurrencyConverterForm()

    return render(
        request,
        "currency_converter.html",
        {"form": form, "result": result, "error_message": error_message},
    )
