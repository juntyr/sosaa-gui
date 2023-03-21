all: gui.py about.py

gui.py: gui.ui
	python3 -m PyQt5.uic.pyuic -x gui.ui -o gui.py

about.py: about.ui
	python3 -m PyQt5.uic.pyuic -x about.ui -o about.py
