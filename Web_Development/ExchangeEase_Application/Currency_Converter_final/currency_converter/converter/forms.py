from django import forms


class CurrencyConverterForm(forms.Form):
    amount = forms.DecimalField(label="Amount", decimal_places=2)
    from_currency = forms.CharField(label="From Currency (e.g. USD)", max_length=3)
    to_currency = forms.CharField(label="To Currency (e.g. EUR)", max_length=3)
