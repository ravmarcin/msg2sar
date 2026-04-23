import unittest
import numpy as np
import datetime
import rasterio
from utils.internal.img.temporal_upsampling import batch_temporal_upsampling


class TestBatchTemporalUpsampling(unittest.TestCase):

    def test_batch_temporal_upsampling(self):
        # Load test data
        path_m = '/Users/raf/Dev/code/msg2sar/tests/unittests/utils/internal/img/test_data/20230315_044441/MSG4-SEVI-MSG15-0100-NA-20230315044244.779000000Z-NA_WV_062.tif'
        path_s = '/Users/raf/Dev/code/msg2sar/tests/unittests/utils/internal/img/test_data/20230315_044441/MSG4-SEVI-MSG15-0100-NA-20230315045745.002000000Z-NA_WV_062.tif'
        
        with rasterio.open(path_m) as src:
            arr_m = src.read(1).astype(float)
        with rasterio.open(path_s) as src:
            arr_s = src.read(1).astype(float)
        
        # Compute time differences (arbitrary for test)
        sev_m_sar_dif = datetime.timedelta(seconds=300)  # 5 minutes
        sev_s_sar_dif = datetime.timedelta(seconds=900)  # 15 minutes
        
        # Parameters
        nd_val = np.nan
        batch_size_x = 100
        batch_size_y = 100
        overlap = 10
        steps_ = [4, 2, 1]
        
        # Call the function
        result = batch_temporal_upsampling(arr_m, arr_s, sev_m_sar_dif, sev_s_sar_dif, nd_val,
                                          batch_size_x, batch_size_y, overlap, steps_,
                                          cut_ref=10, cut_fit=5)
        
        # Write the result image with proper metadata
        output_path = '/Users/raf/Dev/code/msg2sar/tests/unittests/utils/internal/img/test_data/result.tif'
        with rasterio.open(path_m) as src:
            meta = src.meta.copy()
        meta.update(height=result.shape[0], width=result.shape[1], dtype=result.dtype)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(result, 1)
        
        # Load expected result
        with rasterio.open(output_path) as src:
            expected = src.read(1).astype(float)
        
        # Assertions
        self.assertEqual(result.shape, arr_m.shape)
        self.assertFalse(np.all(np.isnan(result)))
        self.assertTrue(np.issubdtype(result.dtype, np.floating))
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


if __name__ == '__main__':
    unittest.main()