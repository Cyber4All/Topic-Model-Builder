import joblib

# Joblib pickels the given model object
# and saves it as a file at the given
# file path
def save_model_to_file(model, file_path):
    joblib.dump(model, file_path)