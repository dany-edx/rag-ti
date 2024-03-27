import pickle
from pathlib import Path
import streamlit_authenticator as stauth

names = ["qcells"]
usernames = ["CTO"]
passwords = ["94107"]

hashed_passwords = stauth.Hasher(passwords).generate()
file_path = Path(__file__).parent / "hased_pw.pkl"
with file_path.open('wb') as file:
    pickle.dump(hased_passwords, file)