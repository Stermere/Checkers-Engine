python -m venv .venv
".venv/Scripts/python.exe" -m pip install -r requirements.txt

cd Marcher_Engine_GUI/client
npm install 
npm run build
cd ....
