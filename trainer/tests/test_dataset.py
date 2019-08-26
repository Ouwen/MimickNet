from trainer import utils

class TestDataset:    
    def test_getpaired(self):
        mimick_dataset = utils.MimickDataset()
        train_dataset, train_count = mimick_dataset.get_paired_ultrasound_dataset(batch_size=1)
        for x,y,z in train_dataset: break
        assert x.shape == y.shape
        
    def test_getpaired_v2(self):
        mimick_dataset = utils.MimickDataset()
        train_dataset, train_count = mimick_dataset.get_paired_ultrasound_dataset(
            csv='gs://duke-research-us/mimicknet/data/training-v2.csv',
            batch_size=1
        )
        for x,y,z in train_dataset: break
        assert x.shape == y.shape

    def test_sc(self):
        mimick_dataset = utils.MimickDataset(sc=True)
        train_dataset, train_count = mimick_dataset.get_paired_ultrasound_dataset(batch_size=1)
        for x,y,z in train_dataset: break
        assert x.shape == y.shape
            
    def test_getunpaired(self):
        a, ac = utils.MimickDataset().get_unpaired_ultrasound_dataset(domain='iq', batch_size=1)
        b, bc = utils.MimickDataset().get_unpaired_ultrasound_dataset(domain='dtce', batch_size=1)
        for x,z1 in a: break
        for y,z2 in b: break
        assert x is not None
        assert y is not None
        assert z1 is not None
        assert z2 is not None

    def test_getunpaired_v2(self):
        a, ac = utils.MimickDataset().get_unpaired_ultrasound_dataset(
            domain='iq', 
            batch_size=1,
            csv='gs://duke-research-us/mimicknet/data/training-v2-verasonics.csv'
        )
        b, bc = utils.MimickDataset().get_unpaired_ultrasound_dataset(
            domain='dtce', 
            batch_size=1,
            csv='gs://duke-research-us/mimicknet/data/training-v2-clinical.csv'
        )
        for x,z1 in a: break
        for y,z2 in b: break
        assert x is not None
        assert y is not None
        assert z1 is not None
        assert z2 is not None
        
    def test_shape(self):
        mimick_dataset = utils.MimickDataset(shape=(256, 256))
        train_dataset, train_count = mimick_dataset.get_paired_ultrasound_dataset(batch_size=1)
        for x,y,z in train_dataset: break
        assert x.shape == (1, 256, 256, 1)
        assert y.shape == (1, 256, 256, 1)
        