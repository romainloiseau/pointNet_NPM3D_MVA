from tensorflow.python.keras.models import model_from_json

def save_model(model, output_path):
    model_json = model.to_json()
    with open(output_path + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(output_path + ".h5")
    print("Saved model to disk")
    
def load_model(input_path):
    json_file = open(input_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(input_path + '.h5')
    print("Loaded model from disk")
    
    return loaded_model
