import h5py
db = h5py.File('synthetic_train_0808_3000000.hdf5', "r")
print(db['labels'][10])