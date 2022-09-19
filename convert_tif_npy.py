# read the dtata from geotif file
import rasterio
import numpy as np
dataset = rasterio.open('/SR_test_data/data1_LiDAR_25cm.tif')
# dataset = rasterio.open('/SR_test_data/data2_LiDAR_25cm.tif')
# dataset = rasterio.open('/SR_test_data/data3_LiDAR_25cm.tif')
# dataset = rasterio.open('/SR_test_data/data1_photo_25cm.tif')
# dataset = rasterio.open('/SR_test_data/data2_photo_25cm.tif')

print('dataset shape = ',dataset.shape)
data = dataset.read(1)
print('data shape = ',data.shape)

np.save('/SR_test_data/data1_LiDAR_25cm.npy',data)
# np.save('/SR_test_data/data2_LiDAR_25cm.npy',data)
# np.save('/SR_test_data/data3_LiDAR_25cm.npy',data)
# np.save('/SR_test_data/data1_photo_25cm.npy',data)
# np.save('/SR_test_data/data2_photo_25cm.npy',data)