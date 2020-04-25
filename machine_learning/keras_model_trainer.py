from machine_learning.constants import BATCH_SIZE, EPOCHS, VALIDATION_SPLIT

# Trains the compiled model on the given topic
# data
def train_model(compiled_model, data, targets):
    compiled_model.fit(
        data,
        targets,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT
    )

    return compiled_model

