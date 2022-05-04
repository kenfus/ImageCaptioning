from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.image as mpimg


# Create Pytorch Dataloader:
class VizWizImageCaptioning(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, preload_images_to_memory=False, transform=None):
        """
        Args:
            df (dataframe): Dataframe containing the caption and URL to the pictures
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform
        self.preload_images_to_memory = preload_images_to_memory
            
        if self.preload_images_to_memory:
            self.images = []
            for url in tqdm(np.unique(df['url'])):
                image = mpimg.imread((url))
                if self.transform:
                    image = self.transform(image)
                self.images.append(image)

            self.df['image_idx'] = df.groupby('url').ngroup()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx, :]
        
        if self.preload_images_to_memory:
            image_id = self.df.iloc[idx]['image_idx']
            image = self.images[image_id]
            
        else:
            image = mpimg.imread(df_row['url'])
            if self.transform:
                image = self.transform(image)


            
        # pack_padded_sequence(torch.tensor(df_row['caption']),  len(df_row['caption']), batch_first=True, enforce_sorted=False)
        return image, np.asarray(df_row['caption']), df_row['length'], df_row['url']

            