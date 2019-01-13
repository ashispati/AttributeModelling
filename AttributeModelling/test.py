from AttributeModelling.data.dataloaders.bar_dataset import*

bar_dataset = FolkNBarDataset(dataset_type='train', is_short=False)
(train_dataloader,
 val_dataloader,
 test_dataloader) = bar_dataset.data_loaders(
    batch_size=128,
    split=(0.7, 0.2)
)
print('Num Train Batches: ', len(train_dataloader))
print('Num Valid Batches: ', len(val_dataloader))
print('Num Test Batches: ', len(test_dataloader))