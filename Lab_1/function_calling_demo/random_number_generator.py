import random
import re
import streamlit as st

def random_number(min, max):
    print("calling the random number function!")

    min = clean_and_convert_to_int(min)
    max = clean_and_convert_to_int(max)
    random_number = random.randint(min, max)
    return random_number

#Cleans and Strip whitespace of Numeric values when expecting integers
def clean_and_convert_to_int(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # Remove spaces and commas
        cleaned = re.sub(r'[,\s]', '', value)
        return int(cleaned)
    raise ValueError(f"Unable to convert {value} to int")
