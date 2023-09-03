from data_utils import find_number_of_characters, create_datasets, generate_batch
import os
import numpy as np
from utils import set_seeds, AdaptiveSchedulerCallback, plot_loss
from models import SimpleModelBuilder
import tensorflow.keras as tfk

# set the seeds
set_seeds()

# get the list of images
captchas = os.listdir('samples')
file_names = list(map(lambda f: f'samples/{f}', captchas)) # add folder name before

# find all characters used in the captchas
chars = find_number_of_characters(captchas)

# create the datasets
IMAGE_SIZE = (50, 200)
chars = np.array(chars)
captchas = list(map(lambda c: c.split('.')[0], captchas)) # remove extenstion from file names
train_data, val_data, test_data = create_datasets(file_names, captchas, chars, image_size = IMAGE_SIZE)

print(train_data.element_spec, len(train_data))
print(val_data.element_spec, len(val_data))
print(test_data.element_spec, len(test_data))

# create the model
builder = SimpleModelBuilder()
model = builder.get_model(input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
model.summary()

# compile the model
model.compile(
    optimizer = tfk.optimizers.SGD(learning_rate = 0.01),
    loss = {
        'out0': tfk.losses.CategoricalCrossentropy(),
        'out1': tfk.losses.CategoricalCrossentropy(),
        'out2': tfk.losses.CategoricalCrossentropy(),
        'out3': tfk.losses.CategoricalCrossentropy(),
        'out4': tfk.losses.CategoricalCrossentropy()
    },
    metrics = {
        'out0': 'accuracy',
        'out1': 'accuracy',
        'out2': 'accuracy',
        'out3': 'accuracy',
        'out4': 'accuracy',
    },
    loss_weights = {
        'out0': 1.0,
        'out1': 1.0,
        'out2': 1.0,
        'out3': 1.0,
        'out4': 1.0,
    }
)



train_gen = generate_batch(train_data, shuffle_buffer_size = len(train_data))
val_gen = generate_batch(val_data, shuffle_buffer_size = len(val_data))
batch_size = 32

history = model.fit(
    x = train_gen,
    validation_data = val_gen,
    epochs = 50,
    batch_size = batch_size,
    steps_per_epoch = len(train_data) // batch_size,
    validation_steps = len(val_data) // batch_size,
    callbacks = [AdaptiveSchedulerCallback()]
)

# plot training loss
plot_loss(history)

# plot validation loss
plot_loss(history, validation = True)