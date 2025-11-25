import google.generativeai as genai
import os
import streamlit as st

# Load the key directly from your secrets file
try:
    import toml
    secrets = toml.load(".streamlit/secrets.toml")
    api_key = secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    
    print("✅ API Key found. Asking Google for available models...\n")
    
    print("--- AVAILABLE MODELS ---")
    found = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model Name: {m.name}")
            found = True
            
    if not found:
        print("❌ No models found. Your API key might be invalid or has no access.")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure you have 'toml' installed: pip install toml")