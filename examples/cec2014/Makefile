PYTHON=python

build:
	$(PYTHON) setup.py build_ext -i
	cp build/lib.linux-x86_64-3.7/cec2014.cpython-37m-x86_64-linux-gnu.so .

clean:
	rm -r build
	rm cec2014.cpython-37m-x86_64-linux-gnu.so
	rm cec2014.cpp
