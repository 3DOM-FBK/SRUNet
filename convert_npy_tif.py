# -*- coding: utf-8 -*-
"""


@author: salim
"""

import rasterio
import numpy as np
dataset = rasterio.open('/SR_test_data/data1_LiDAR_25cm.tif')
# dataset = rasterio.open('/SR_test_data/data2_LiDAR_25cm.tif')
# dataset = rasterio.open('/SR_test_data/data3_LiDAR_25cm.tif')
# dataset = rasterio.open('/SR_test_data/data1_photo_25cm.tif')
# dataset = rasterio.open('/SR_test_data/data2_photo_25cm.tif')

print('dataset shape = ',dataset.shape)
ras_meta=dataset.profile
print(ras_meta)


ima=np.load('/SR_test_data/data1_LiDAR_res.npy')
# ima=np.load('/SR_test_data/data2_LiDAR_res.npy')
# ima=np.load('/SR_test_data/data3_LiDAR_res.npy')
# ima=np.load('/SR_test_data/data1_photo_res.npy')
# ima=np.load('/SR_test_data/data2_photo_res.npy')



print('ima npy shape = ',ima.shape)

    
with rasterio.open('/SR_test_data/data1_LiDAR_res.tif', 'w', **ras_meta) as dst:
    dst.write(ima,indexes=1)
# with rasterio.open('/SR_test_data/data2_LiDAR_res.tif', 'w', **ras_meta) as dst:
#     dst.write(ima,indexes=1)
# with rasterio.open('/SR_test_data/data3_LiDAR_res.tif', 'w', **ras_meta) as dst:
#     dst.write(ima,indexes=1)
# with rasterio.open('/SR_test_data/data1_photo_res.tif', 'w', **ras_meta) as dst:
#     dst.write(ima,indexes=1)
# with rasterio.open('/SR_test_data/data2_photo_res.tif', 'w', **ras_meta) as dst:
#     dst.write(ima,indexes=1)