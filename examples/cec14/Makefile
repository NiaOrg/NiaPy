PYTHON=python

build:
	$(PYTHON) setup.py build_ext -i
	cp build/lib.linux-x86_64-3.7/function_call.cpython-37m-x86_64-linux-gnu.so .

run:
	$(PYTHON) -c 'from function_call import run_fun; import numpy as np; print(run_fun(np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.double), 1))'
	$(PYTHON) -c 'from function_call import run_fun; import numpy as np; print(run_fun(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.double), 1))'
	$(PYTHON) -c 'from function_call import run_fun; import numpy as np; print(run_fun(np.asarray([1, 2, 42, 4, 5, 12, 7, 8, 23, 10], dtype=np.double), 1)); print(run_fun(np.asarray([1, 1, 3, 1, 5, 6, 7, 8, 9, 4], dtype=np.double), 1))'

clean:
	rm -r build
	rm function_call.cpython-37m-x86_64-linux-gnu.so
