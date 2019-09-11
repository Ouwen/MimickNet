import pandas as pd
import numpy as np

# This file is used to split training, validation, and test csv by target

df = pd.read_csv('master.csv')

# These images are well laterally sampled and used for publication
test_timestamp = np.array([
        20121101150345,
        20121031150931,                        
        20161216093626,
        20180206194115,
        20170830145820,
        20160909160351,
        20160901073342
])

timestamps = np.array((df[(~df['timestamp_id'].isin(test_timestamp))].groupby('timestamp_id').first().index))
np.random.shuffle(timestamps)
a = np.concatenate((timestamps[:393], test_timestamp), axis=0)
b = timestamps[493:]
c = timestamps[393:493]

testing_df = df[df['timestamp_id'].isin(a)]
training_df = df[df['timestamp_id'].isin(b)]
validation_df = df[df['timestamp_id'].isin(c)]

testing_df.to_csv('testing-v2.csv', index=False)
training_df.to_csv('training-v2.csv', index=False)
validation_df.to_csv('validation-v2.csv', index=False)

verasonics = training_df[(training_df['scanner']=='verasonics')]
not_verasonics = training_df[(training_df['scanner']!='verasonics')]

verasonics.to_csv('training-v2-verasonics.csv', index=False)
not_verasonics.to_csv('training-v2-clinical.csv', index=False)
