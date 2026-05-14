import unittest
import os
import tempfile
import shutil
from datetime import date
from unittest.mock import patch, MagicMock

from libs.internal.gnss.rinex.orbit_downloader import (
    _gps_week_day,
    _sp3_filename,
    _clk_filename,
    download_orbits,
    download_orbits_for_range,
)


class TestGpsWeekDay(unittest.TestCase):

    def test_gps_epoch(self):
        # GPS epoch: 1980-01-06 is week 0, day 0 (Sunday)
        week, day = _gps_week_day(date(1980, 1, 6))
        self.assertEqual(week, 0)
        self.assertEqual(day, 0)

    def test_known_date(self):
        # 2021-04-10 is GPS week 2152, day 6 (Saturday)
        week, day = _gps_week_day(date(2021, 4, 10))
        self.assertEqual(week, 2152)
        self.assertEqual(day, 6)

    def test_known_date_sunday(self):
        # 2021-04-11 is GPS week 2153, day 0 (Sunday)
        week, day = _gps_week_day(date(2021, 4, 11))
        self.assertEqual(week, 2153)
        self.assertEqual(day, 0)

    def test_known_date_monday(self):
        # 2021-04-12 is GPS week 2153, day 1 (Monday)
        week, day = _gps_week_day(date(2021, 4, 12))
        self.assertEqual(week, 2153)
        self.assertEqual(day, 1)


class TestFilenameGeneration(unittest.TestCase):

    def test_sp3_filename_default_product(self):
        result = _sp3_filename(date(2021, 4, 10))
        self.assertEqual(result, "gfz21526.sp3")

    def test_sp3_filename_custom_product(self):
        result = _sp3_filename(date(2021, 4, 10), product="igs")
        self.assertEqual(result, "igs21526.sp3")

    def test_clk_filename_default_product(self):
        result = _clk_filename(date(2021, 4, 10))
        self.assertEqual(result, "gfz21526.clk")

    def test_clk_filename_custom_product(self):
        result = _clk_filename(date(2021, 4, 10), product="cod")
        self.assertEqual(result, "cod21526.clk")

    def test_sp3_filename_next_day(self):
        # 2021-04-11 is week 2153, day 0
        result = _sp3_filename(date(2021, 4, 11))
        self.assertEqual(result, "gfz21530.sp3")


class TestDownloadOrbits(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("libs.internal.gnss.rinex.orbit_downloader._download_file")
    @patch("libs.internal.gnss.rinex.orbit_downloader._decompress_z")
    def test_download_orbits_success(self, mock_decompress, mock_download):
        epoch = date(2021, 4, 10)

        def fake_download(url, output_path, overwrite=False):
            with open(output_path, "w") as f:
                f.write("data")
            return True

        def fake_decompress(z_path):
            out_path = z_path[:-2]
            with open(out_path, "w") as f:
                f.write("decompressed")
            return out_path

        mock_download.side_effect = fake_download
        mock_decompress.side_effect = fake_decompress

        result = download_orbits(epoch, self.tmpdir, product="gfz")

        self.assertIn("sp3_yesterday", result)
        self.assertIn("sp3_today", result)
        self.assertIn("sp3_tomorrow", result)
        self.assertIn("clk_today", result)

        # All should be non-None
        for key, val in result.items():
            self.assertIsNotNone(val, f"{key} should not be None")

        # Should have called download 4 times
        self.assertEqual(mock_download.call_count, 4)

    @patch("libs.internal.gnss.rinex.orbit_downloader._download_file")
    def test_download_orbits_skips_existing(self, mock_download):
        epoch = date(2021, 4, 10)

        # Pre-create all expected decompressed files
        expected_files = [
            _sp3_filename(date(2021, 4, 9)),
            _sp3_filename(date(2021, 4, 10)),
            _sp3_filename(date(2021, 4, 11)),
            _clk_filename(date(2021, 4, 10)),
        ]
        for fname in expected_files:
            with open(os.path.join(self.tmpdir, fname), "w") as f:
                f.write("existing")

        result = download_orbits(epoch, self.tmpdir, product="gfz")

        # Should not have attempted any downloads
        mock_download.assert_not_called()

        # All results should point to existing files
        for key, val in result.items():
            self.assertIsNotNone(val)
            self.assertTrue(os.path.exists(val))

    @patch("libs.internal.gnss.rinex.orbit_downloader._download_file")
    def test_download_orbits_handles_failure(self, mock_download):
        epoch = date(2021, 4, 10)
        mock_download.return_value = False

        result = download_orbits(epoch, self.tmpdir, product="gfz")

        # All should be None since downloads failed
        for key, val in result.items():
            self.assertIsNone(val, f"{key} should be None on download failure")

    def test_download_orbits_creates_output_dir(self):
        new_dir = os.path.join(self.tmpdir, "new_subdir", "orbits")
        self.assertFalse(os.path.exists(new_dir))

        with patch("libs.internal.gnss.rinex.orbit_downloader._download_file", return_value=False):
            download_orbits(date(2021, 4, 10), new_dir)

        self.assertTrue(os.path.isdir(new_dir))


class TestDownloadOrbitsForRange(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("libs.internal.gnss.rinex.orbit_downloader.download_orbits")
    def test_range_calls_download_for_each_day(self, mock_download):
        mock_download.return_value = {
            "sp3_yesterday": "/fake/a.sp3",
            "sp3_today": "/fake/b.sp3",
            "sp3_tomorrow": "/fake/c.sp3",
            "clk_today": "/fake/b.clk",
        }

        result = download_orbits_for_range(
            date(2021, 4, 10), date(2021, 4, 12), self.tmpdir
        )

        # Should call download_orbits for Apr 10, 11, 12 = 3 calls
        self.assertEqual(mock_download.call_count, 3)

        # Result should be deduplicated sorted list
        self.assertIsInstance(result, list)

    @patch("libs.internal.gnss.rinex.orbit_downloader.download_orbits")
    def test_range_single_day(self, mock_download):
        mock_download.return_value = {
            "sp3_yesterday": "/fake/a.sp3",
            "sp3_today": "/fake/b.sp3",
            "sp3_tomorrow": "/fake/c.sp3",
            "clk_today": "/fake/b.clk",
        }

        download_orbits_for_range(date(2021, 4, 10), date(2021, 4, 10), self.tmpdir)

        mock_download.assert_called_once()

    @patch("libs.internal.gnss.rinex.orbit_downloader.download_orbits")
    def test_range_filters_none_values(self, mock_download):
        mock_download.return_value = {
            "sp3_yesterday": "/fake/a.sp3",
            "sp3_today": None,
            "sp3_tomorrow": None,
            "clk_today": "/fake/b.clk",
        }

        result = download_orbits_for_range(
            date(2021, 4, 10), date(2021, 4, 10), self.tmpdir
        )

        self.assertNotIn(None, result)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
