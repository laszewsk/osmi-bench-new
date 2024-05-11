BASE=../../..

clean:
	rm -rf build

setup-models:
	mkdir -p build/tfserving/archs
	cp -r ${BASE}/tfserving/archs build/tfserving

images:
	cd train; make iamge

