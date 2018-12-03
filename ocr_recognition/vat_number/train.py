from keras.optimizers import SGD
from ocr_recognition.vat_number.lenet import LeNet
from ocr_recognition.vat_number.hdf5 import HDF5DatasetGenerator
from ocr_recognition.common.callback import *
from keras import callbacks

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
model = LeNet.build(width=28, height=28, depth=1, num_classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# callbacks
training_monitor = TrainingMonitor(figure_path='vat_number_1129_5000_600.jpg',
                                   json_path='vat_number_1129_5000_600.json',
                                   start_at=0)
# accuracy_evaluator = AccuracyEvaluator(TEST_DB_PATH, batch_size=100)
learning_rate_updator = LearningRateUpdator(init_lr=0.001)
callbacks = [
    # Interrupts training when improvement stops
    callbacks.EarlyStopping(
        # Monitors the model’s validation accuracy
        monitor='val_loss',
        # Interrupts training when accuracy has stopped
        # improving for more than one epoch (that is, two epochs)
        patience=10,
    ),
    # Saves the current weights after every epoch
    callbacks.ModelCheckpoint(
        # Path to the destination model file
        filepath='models/vat_number_1129_5000_600.hdf5',
        # These two arguments mean you won’t overwrite the
        # model file unless val_loss has improved, which allows
        # you to keep the best model seen during training.
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    training_monitor,
    # accuracy_evaluator
    # learning_rate_updator
]
gen = HDF5DatasetGenerator('data/vat_number_train_1130_6000.hdf5', batch_size=200).generator
val_gen = HDF5DatasetGenerator('data/vat_number_val_1130_2000.hdf5', batch_size=100).generator
H = model.fit_generator(gen(), steps_per_epoch=30,
                        callbacks=callbacks,
                        epochs=100,
                        validation_data=val_gen(),
                        validation_steps=20)
N = np.arange(0, len(H.history["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
