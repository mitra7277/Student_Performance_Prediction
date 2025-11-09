import streamlit as st
import pandas as pd
import hashlib
import os

USERS_FILE = "users.csv"

# -----------------------------
# Helper: Hash Password
# -----------------------------
def hash_pass(password):
    return hashlib.sha256(password.encode()).hexdigest()

# -----------------------------
# Helper: Load Users
# -----------------------------
def load_users():
    if os.path.exists(USERS_FILE):
        return pd.read_csv(USERS_FILE)
    else:
        return pd.DataFrame(columns=["Username", "Password"])

# -----------------------------
# Helper: Save Users
# -----------------------------
def save_users(df):
    df.to_csv(USERS_FILE, index=False)

# -----------------------------
# Register New User
# -----------------------------
def login_user(username, password):
    import pandas as pd
    users = pd.read_csv("users.csv")
    # Match lowercase columns
    if username in users["username"].values and users.loc[users["username"]==username, "password"].values[0] == password:
        return True
    return False

def register_user(username, password):
    import pandas as pd
    users = pd.read_csv("users.csv") if os.path.getsize("users.csv")>0 else pd.DataFrame(columns=["username","password"])
    if username in users["username"].values:
        return False
    new_row = pd.DataFrame([[username,password]], columns=["username","password"])
    new_row.to_csv("users.csv", mode='a', header=not os.path.exists("users.csv") or os.path.getsize("users.csv")==0, index=False)
    return True
