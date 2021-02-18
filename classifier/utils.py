from classifier import Brain, DataWrapper


def train_brain(brain:Brain, dataset_name:str)->Brain:
    """Retrain a brain on the given dataset

    Parameters
    ----------
    brain : Brain
        A brain object to retrain.
    dataset_name : str
        The dataset name, must be one that the DataWrapper class supports.

    Returns
    -------
    Brain
        A trained Brain on the given dataset.
    """
    train_set, test_set = DataWrapper.get_datasets()[dataset_name]()
    brain.fit(train_set, DataWrapper.get_input_channels()[dataset_name])
    brain.export_model()
    return brain