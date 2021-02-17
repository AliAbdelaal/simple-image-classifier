from classifier import Brain, Classifier, DataWrapper


def train_brain(brain:Brain, dataset_name:str):
    train_set, test_set = DataWrapper.get_datasets()[dataset_name]()
    brain.fit(train_set, DataWrapper.get_input_channels()[dataset_name])
    brain.export_model()
    return brain