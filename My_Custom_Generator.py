class My_Custom_Generator(keras.utils.Sequence):
    def __init__(self, data_x, data_y, batch_size):
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
    
    def __getsingleitem__(self, idx):
        batch_x = self.data_x[idx * self.batch_size : (idx+1) * self.batch_size]
        return np.array(batch_x)

    def __getitem__(self, idx):
        batch_x = self.data_x[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.data_y[idx * self.batch_size : (idx+1) * self.batch_size]
        return np.array(batch_x), np.array(batch_y)