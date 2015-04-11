all: hdf5 nngraph eventdetection

.PHONY: hdf5
hdf5:
	sudo apt-get install libhdf5-serial-dev hdf5-tools
	git clone https://github.com/deepmind/torch-hdf5.git
	cd torch-hdf5 && luarocks make hdf5-0-0.rockspec

.PHONY: nngraph
nngraph:
	luarocks install nngraph

.PHONY: eventdetection
eventdetection:
	git clone https://github.com/pabloi/dataDriven-eventDetection
