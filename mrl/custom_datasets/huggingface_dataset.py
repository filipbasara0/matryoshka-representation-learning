from torch.utils.data import Dataset


class HuggingfaceDataset(Dataset):

    def __init__(self, data, transform, image_key="image", label_key="label"):
        super(HuggingfaceDataset, self).__init__()
        self.data = data
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx][self.image_key], int(
            self.data[idx][self.label_key])

        image = self.transform(image)
        return image, label
