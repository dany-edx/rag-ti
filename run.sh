#!/bin/bash



export DISPLAY=:0.0



# 첫 번째 프로세스 실행

gnome-terminal --tab --title="Streamlit" -- bash -c 'streamlit run ./ui/main.py; $SHELL'



# 두 번째 프로세스 실행

gnome-terminal --tab --title="Jupyter Lab" -- bash -c 'jupyter lab --ip="*" --port=5678 --no-browser --allow-root; $SHELL'



# 세 번째 프로세스 실행

gnome-terminal --tab --title="Chroma" -- bash -c 'chroma run --path ./db/pvmagazine_db/ --port 8000 --host 0.0.0.0; $SHELL'


