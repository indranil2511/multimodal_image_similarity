import streamlit as st
from layout import load
from chat import main

def run():
    load()
    main()



from app import run

if __name__ == "__main__":
    run()