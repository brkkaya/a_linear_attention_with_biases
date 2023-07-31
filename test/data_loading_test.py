from dataset_loading.data_loading import data_loading_hf_dataset, data_loading_custom_dataset

def test_hf_loading():
    train_loader, test_loader = data_loading_hf_dataset(
        "databricks/databricks-dolly-15k", "mosaicml/mpt-7b", test_size=2000
    )
    for sample in train_loader:
        print(sample)
        break

def test_custom_loading():
    train_loader, test_loader = data_loading_custom_dataset(
        "data/alpaca_cleaned.json", "mosaicml/mpt-7b", test_size=2000
    )
    for sample in train_loader:
        print(sample)
        break

test_custom_loading()
