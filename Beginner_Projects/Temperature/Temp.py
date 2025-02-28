import streamlit as st

# Conversion functions
def celsius_to_fahrenheit(c):
    return (c * 9/5) + 32

def celsius_to_kelvin(c):
    return c + 273.15

def fahrenheit_to_celsius(f):
    return (f - 32) * 5/9

def fahrenheit_to_kelvin(f):
    return celsius_to_kelvin(fahrenheit_to_celsius(f))

def kelvin_to_celsius(k):
    return k - 273.15

def kelvin_to_fahrenheit(k):
    return celsius_to_fahrenheit(kelvin_to_celsius(k))

# Streamlit app
st.title("Temperature Converter")

# Input widgets
unit = st.selectbox("Select Unit:", ['Celsius', 'Fahrenheit', 'Kelvin'])
temp = st.number_input("Temperature:", value=0.0)

# Convert button
if st.button("Convert"):
    if unit == 'Celsius':
        fahrenheit = celsius_to_fahrenheit(temp)
        kelvin = celsius_to_kelvin(temp)
        st.success(f"{temp:.2f} °C = {fahrenheit:.2f} °F = {kelvin:.2f} K")
    elif unit == 'Fahrenheit':
        celsius = fahrenheit_to_celsius(temp)
        kelvin = fahrenheit_to_kelvin(temp)
        st.success(f"{temp:.2f} °F = {celsius:.2f} °C = {kelvin:.2f} K")
    elif unit == 'Kelvin':
        celsius = kelvin_to_celsius(temp)
        fahrenheit = kelvin_to_fahrenheit(temp)
        st.success(f"{temp:.2f} K = {celsius:.2f} °C = {fahrenheit:.2f} °F")