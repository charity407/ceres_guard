import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import requests

print("✅ Pandas Version:", pd.__version__)
print("✅ NumPy Version:", np.__version__)
print("✅ Scikit-Learn is ready!")
print("✅ Streamlit is ready!")

# Test if the Telegram logic works (Optional)
# Uncomment the lines below and add your details to test your phone alert
# TOKEN = "YOUR_BOT_TOKEN"
# CHAT_ID = "YOUR_CHAT_ID"
# url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text=GrainGuard System Online 🌾"
# requests.get(url)