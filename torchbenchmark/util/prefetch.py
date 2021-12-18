

def get_data_example(loader):
    "Get the first batch in the dataloader or prefeteched data array"
    assert len(loader) > 0, "The dataloader or data array is empty."
    for data in loader:
        return data

def prefetch_loader(loader, device):
    result = []
    for data in loader:
        items = []
        for item in data:
            items.append(item.to(device))
        result.append(tuple(items))
    return result