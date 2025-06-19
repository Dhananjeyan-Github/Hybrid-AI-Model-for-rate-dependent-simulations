import numpy as np

class prep():
    def __init__(self, fdir, n_trajectories = None, seed = 42):
        super(prep, self).__init__()
        np.random.seed(seed)
        self.fdir = fdir
        self.n_samples = n_trajectories
        
    def train_test_sp(self, data):

        sample_size = int(0.2 * data[0].shape[0])
        self.test_indices = np.random.choice(data[0].shape[0], sample_size, replace=False)
        data_test = data[0][self.test_indices]
        self.train_indices = np.setdiff1d(np.arange(data[0].shape[0]), self.test_indices)
        data_train = data[0][self.train_indices]
        data_val = data[1]

        return (data_train, data_test, data_val)
        
    def load(self):

        zdata = np.load(self.fdir, allow_pickle=True)
        data = zdata['data']
        data_train_test= data[:self.n_samples, : ,1:]
        data_val = data[self.n_samples:, :, 1:]

        return [data_train_test, data_val]
    
    
    def scale(self):

        data_train, data_test, data_val = self.train_test_sp(self.load())
        mean_bc = np.mean(data_train, axis=(0,1))
        std_bc = np.std(data_train)
        data_train = (data_train - mean_bc) / std_bc
        data_test = (data_test - mean_bc) / std_bc
        data_val = (data_val - mean_bc) / std_bc
        data_train = data_train.reshape(data_train.shape[0], 10, -1, order='F')
        data_test = data_test.reshape(data_test.shape[0], 10, -1, order='F')
        data_val = data_val.reshape(data_val.shape[0], 10, -1, order='F')

        return (data_train, data_test, data_val, mean_bc, std_bc)
    

    def load_local(self, fdir):

        zdata = np.load(fdir, allow_pickle=True)
        data = zdata['data']
        train = data[self.train_indices]
        test = data[self.test_indices]
        val = data[self.n_samples:,:,:,:]
        data_train = train.reshape((train.shape[0], train.shape[1], -1), order='F')
        data_test = test.reshape((test.shape[0], test.shape[1], -1), order='F')
        data_val = val.reshape((val.shape[0],val.shape[1], -1), order='F')
        mean_local = np.mean(data_train, axis=(0,1))
        std_local = np.std(data_train)
        data_train = (data_train - mean_local) / std_local
        data_test = (data_test - mean_local) / std_local
        data_val = (data_val - mean_local) / std_local

        return (data_train, data_test, data_val, mean_local, std_local)
    
    def load_local_prev(self, fdir):

        zdata = np.load(fdir, allow_pickle=True)
        data = zdata['data']
        self.train_indices_prev = self.train_indices-1
        self.test_indices_prev = self.test_indices-1
        train = data[self.train_indices_prev]
        test = data[self.test_indices_prev]
        val = data[self.n_samples:,:,:,:]
        data_train = train.reshape((train.shape[0], train.shape[1], -1), order='F')
        data_test = test.reshape((test.shape[0], test.shape[1], -1), order='F')
        data_val = val.reshape((val.shape[0],val.shape[1], -1), order='F')
        mean_local = np.mean(data_train, axis=(0,1))
        std_local = np.std(data_train)
        data_train = (data_train - mean_local) / std_local
        data_test = (data_test - mean_local) / std_local
        data_val = (data_val - mean_local) / std_local

        return (data_train, data_test, data_val, mean_local, std_local)
        

    def load_time(self, fdir):

        zdata = np.load(fdir, allow_pickle=True)
        data = zdata['data']
        data=data[:,:,0]
        train = data[self.train_indices]
        test = data[self.test_indices]
        val = data[self.n_samples:,:]

        return (train, test, val)
    

    def load_micro(self, fdir):

        zdata = np.load(fdir, allow_pickle=True)
        data = zdata['data']
        train = data[self.train_indices]
        test = data[self.test_indices]
        val = data[self.n_samples:,:,:,:]

        return (train, test, val)
    
    def load_global(self, fdir):

        zdata = np.load(fdir, allow_pickle=True)
        data = zdata['data']
        train = data[self.train_indices]
        test = data[self.test_indices]
        val = data[self.n_samples:,:,:]

        return (train, test, val)
    

    
    


    

    

    



    

    
    
