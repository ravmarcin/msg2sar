import unittest
import os
import gzip
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock

from libs.internal.gnss.rinex.rinex_processor import GnssRinexProcessor


class TestGnssRinexProcessorInit(unittest.TestCase):

    def test_valid_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = GnssRinexProcessor(tmpdir)
            self.assertEqual(processor.rinex_dir, tmpdir)

    def test_invalid_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            GnssRinexProcessor("/nonexistent/path/to/rinex")

    def test_empty_caches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = GnssRinexProcessor(tmpdir)
            self.assertEqual(processor._observations, {})
            self.assertEqual(processor._navigations, {})


class TestDecompressGz(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_decompress_gz_file(self):
        gz_path = os.path.join(self.tmpdir, "test.rnx.gz")
        content = b"RINEX test content"
        with gzip.open(gz_path, "wb") as f:
            f.write(content)

        result = GnssRinexProcessor._decompress_gz(gz_path)

        self.assertEqual(result, gz_path[:-3])
        self.assertTrue(os.path.exists(result))
        with open(result, "rb") as f:
            self.assertEqual(f.read(), content)

    def test_skip_if_already_decompressed(self):
        gz_path = os.path.join(self.tmpdir, "test.rnx.gz")
        out_path = os.path.join(self.tmpdir, "test.rnx")

        # Create the decompressed file first
        with open(out_path, "wb") as f:
            f.write(b"already decompressed")

        # Create the gz file with different content
        with gzip.open(gz_path, "wb") as f:
            f.write(b"compressed content")

        result = GnssRinexProcessor._decompress_gz(gz_path)

        self.assertEqual(result, out_path)
        # Should keep the original decompressed file, not overwrite
        with open(result, "rb") as f:
            self.assertEqual(f.read(), b"already decompressed")

    def test_non_gz_file_returned_as_is(self):
        filepath = os.path.join(self.tmpdir, "test.rnx")
        result = GnssRinexProcessor._decompress_gz(filepath)
        self.assertEqual(result, filepath)


class TestCrxToRnx(unittest.TestCase):

    def test_non_crx_file_returned_as_is(self):
        result = GnssRinexProcessor._crx_to_rnx("/some/path/test.rnx")
        self.assertEqual(result, "/some/path/test.rnx")

    @patch("hatanaka.decompress")
    def test_successful_conversion(self, mock_decompress):
        with tempfile.TemporaryDirectory() as tmpdir:
            crx_path = os.path.join(tmpdir, "test.crx")
            rnx_path = os.path.join(tmpdir, "test.rnx")

            with open(crx_path, "wb") as f:
                f.write(b"crx data")

            mock_decompress.return_value = b"rnx data"

            result = GnssRinexProcessor._crx_to_rnx(crx_path)
            self.assertEqual(result, rnx_path)
            self.assertTrue(os.path.exists(rnx_path))
            with open(rnx_path, "rb") as f:
                self.assertEqual(f.read(), b"rnx data")

    @patch("hatanaka.decompress", side_effect=Exception("conversion failed"))
    @patch("gnsspy.crx2rnx", side_effect=Exception("binary failed"))
    def test_conversion_failure_returns_crx(self, mock_gnsspy, mock_hatanaka):
        with tempfile.TemporaryDirectory() as tmpdir:
            crx_path = os.path.join(tmpdir, "test.crx")
            with open(crx_path, "wb") as f:
                f.write(b"crx data")

            result = GnssRinexProcessor._crx_to_rnx(crx_path)
            self.assertEqual(result, crx_path)

    def test_skip_if_rnx_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            crx_path = os.path.join(tmpdir, "test.crx")
            rnx_path = os.path.join(tmpdir, "test.rnx")

            with open(crx_path, "w") as f:
                f.write("crx")
            with open(rnx_path, "w") as f:
                f.write("rnx already exists")

            result = GnssRinexProcessor._crx_to_rnx(crx_path)
            self.assertEqual(result, rnx_path)


class TestPrepareFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processor = GnssRinexProcessor(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_prepare_plain_rnx(self):
        filepath = os.path.join(self.tmpdir, "test.rnx")
        with open(filepath, "w") as f:
            f.write("rnx data")

        result = self.processor._prepare_file("test.rnx")
        self.assertEqual(result, filepath)

    def test_prepare_gz_file(self):
        gz_name = "test.rnx.gz"
        gz_path = os.path.join(self.tmpdir, gz_name)
        with gzip.open(gz_path, "wb") as f:
            f.write(b"rnx content")

        result = self.processor._prepare_file(gz_name)

        expected = os.path.join(self.tmpdir, "test.rnx")
        self.assertEqual(result, expected)
        self.assertTrue(os.path.exists(expected))

    @patch.object(GnssRinexProcessor, "_crx_to_rnx")
    def test_prepare_crx_gz_file(self, mock_crx_to_rnx):
        gz_name = "test.crx.gz"
        gz_path = os.path.join(self.tmpdir, gz_name)
        crx_path = os.path.join(self.tmpdir, "test.crx")

        with gzip.open(gz_path, "wb") as f:
            f.write(b"crx content")

        mock_crx_to_rnx.return_value = os.path.join(self.tmpdir, "test.rnx")

        result = self.processor._prepare_file(gz_name)

        mock_crx_to_rnx.assert_called_once_with(crx_path)
        self.assertEqual(result, os.path.join(self.tmpdir, "test.rnx"))


class TestReadObservation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processor = GnssRinexProcessor(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("gnsspy.read_obsFile")
    @patch.object(GnssRinexProcessor, "_prepare_file")
    def test_read_observation_success(self, mock_prepare, mock_read_obs):
        mock_prepare.return_value = os.path.join(self.tmpdir, "test.rnx")

        # Build a mock Observation object
        mock_obs = MagicMock()
        mock_obs.interval = 30
        mock_obs.version = 3.04

        import pandas as pd
        index = pd.MultiIndex.from_tuples(
            [("2021-04-10 00:00:00", "G01"), ("2021-04-10 00:00:00", "G02"),
             ("2021-04-10 00:00:30", "G01"), ("2021-04-10 00:00:30", "G02")],
            names=["Epoch", "SV"]
        )
        mock_obs.observation = pd.DataFrame({"C1C": [1, 2, 3, 4]}, index=index)
        mock_read_obs.return_value = mock_obs

        result = self.processor.read_observation("test.rnx")

        self.assertEqual(result, mock_obs)
        mock_read_obs.assert_called_once()

    @patch("gnsspy.read_obsFile")
    @patch.object(GnssRinexProcessor, "_prepare_file")
    def test_read_observation_caches(self, mock_prepare, mock_read_obs):
        mock_prepare.return_value = os.path.join(self.tmpdir, "test.rnx")

        mock_obs = MagicMock()
        mock_obs.interval = 30
        mock_obs.version = 3.04
        import pandas as pd
        index = pd.MultiIndex.from_tuples(
            [("2021-04-10", "G01")], names=["Epoch", "SV"]
        )
        mock_obs.observation = pd.DataFrame({"C1C": [1]}, index=index)
        mock_read_obs.return_value = mock_obs

        # First call
        self.processor.read_observation("test.rnx")
        # Second call should use cache
        result = self.processor.read_observation("test.rnx")

        self.assertEqual(result, mock_obs)
        mock_read_obs.assert_called_once()


class TestReadNavigation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processor = GnssRinexProcessor(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("gnsspy.read_navFile")
    @patch.object(GnssRinexProcessor, "_prepare_file")
    def test_read_navigation_success(self, mock_prepare, mock_read_nav):
        mock_prepare.return_value = os.path.join(self.tmpdir, "test_GN.rnx")

        import pandas as pd
        mock_nav = pd.DataFrame({"a0": [1.0, 2.0], "e": [0.01, 0.02]})
        mock_read_nav.return_value = mock_nav

        result = self.processor.read_navigation("test_GN.rnx")

        self.assertEqual(len(result), 2)
        mock_read_nav.assert_called_once()

    @patch("gnsspy.read_navFile")
    @patch.object(GnssRinexProcessor, "_prepare_file")
    def test_read_navigation_caches(self, mock_prepare, mock_read_nav):
        mock_prepare.return_value = os.path.join(self.tmpdir, "test_GN.rnx")

        import pandas as pd
        mock_nav = pd.DataFrame({"a0": [1.0]})
        mock_read_nav.return_value = mock_nav

        self.processor.read_navigation("test_GN.rnx")
        self.processor.read_navigation("test_GN.rnx")

        mock_read_nav.assert_called_once()


class TestComputeSpp(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processor = GnssRinexProcessor(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("gnsspy.spp")
    @patch.object(GnssRinexProcessor, "read_observation")
    def test_compute_spp(self, mock_read_obs, mock_spp):
        mock_station = MagicMock()
        mock_read_obs.return_value = mock_station

        import pandas as pd
        mock_position = pd.DataFrame({
            "X": [3655335.0], "Y": [1403940.0], "Z": [5018005.0],
            "receiver_clock": [0.001],
        })
        mock_spp.return_value = mock_position

        orbit = MagicMock()
        result = self.processor.compute_spp("test.rnx", orbit, system="G", cut_off=7.0)

        self.assertEqual(len(result), 1)
        mock_spp.assert_called_once_with(mock_station, orbit, system="G", cut_off=7.0)

    @patch("gnsspy.spp")
    @patch.object(GnssRinexProcessor, "read_observation")
    def test_compute_spp_default_params(self, mock_read_obs, mock_spp):
        mock_read_obs.return_value = MagicMock()

        import pandas as pd
        mock_spp.return_value = pd.DataFrame({"X": [1.0]})

        orbit = MagicMock()
        self.processor.compute_spp("test.rnx", orbit)

        mock_spp.assert_called_once_with(
            mock_read_obs.return_value, orbit, system="G", cut_off=7.0
        )


class TestComputeMultipath(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processor = GnssRinexProcessor(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("gnsspy.multipath")
    @patch.object(GnssRinexProcessor, "read_observation")
    def test_compute_multipath(self, mock_read_obs, mock_mp):
        mock_station = MagicMock()
        mock_read_obs.return_value = mock_station

        import pandas as pd
        mock_mp.return_value = pd.DataFrame({
            "Multipath1": [0.5, 0.3], "Multipath2": [0.4, 0.2]
        })

        result = self.processor.compute_multipath("test.rnx", system="G")

        self.assertEqual(len(result), 2)
        mock_mp.assert_called_once_with(mock_station, system="G")


class TestComputeTroposphericDelay(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processor = GnssRinexProcessor(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("gnsspy.tropospheric_delay")
    @patch.object(GnssRinexProcessor, "read_observation")
    def test_compute_tropo_delay(self, mock_read_obs, mock_tropo):
        mock_station = MagicMock()
        mock_station.approx_position = [3655335.0, 1403940.0, 5018005.0]
        mock_station.epoch = datetime(2021, 4, 10)
        mock_read_obs.return_value = mock_station

        mock_tropo.return_value = 2.3456

        result = self.processor.compute_tropospheric_delay("test.rnx", elevation=90.0)

        self.assertAlmostEqual(result, 2.3456)
        mock_tropo.assert_called_once_with(
            3655335.0, 1403940.0, 5018005.0, 90.0, datetime(2021, 4, 10)
        )

    @patch("gnsspy.tropospheric_delay")
    @patch.object(GnssRinexProcessor, "read_observation")
    def test_compute_tropo_delay_default_elevation(self, mock_read_obs, mock_tropo):
        mock_station = MagicMock()
        mock_station.approx_position = [1.0, 2.0, 3.0]
        mock_station.epoch = datetime(2021, 4, 10)
        mock_read_obs.return_value = mock_station

        mock_tropo.return_value = 2.0

        self.processor.compute_tropospheric_delay("test.rnx")

        # Default elevation should be 90.0
        mock_tropo.assert_called_once_with(1.0, 2.0, 3.0, 90.0, datetime(2021, 4, 10))


class TestInterpolateOrbits(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processor = GnssRinexProcessor(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("gnsspy.sp3_interp")
    def test_interpolate_orbits(self, mock_sp3):
        import pandas as pd
        index = pd.MultiIndex.from_tuples(
            [("G01", "2021-04-10 00:00:00"), ("G01", "2021-04-10 00:00:30"),
             ("G02", "2021-04-10 00:00:00"), ("G02", "2021-04-10 00:00:30")],
            names=["SV", "Epoch"]
        )
        mock_orbit = pd.DataFrame({
            "X": [1, 2, 3, 4], "Y": [5, 6, 7, 8], "Z": [9, 10, 11, 12]
        }, index=index)
        mock_sp3.return_value = mock_orbit

        epoch = datetime(2021, 4, 10)
        result = self.processor.interpolate_orbits(epoch, interval=30, sp3_product="gfz")

        self.assertEqual(len(result), 4)
        mock_sp3.assert_called_once_with(
            epoch, interval=30, sp3_product="gfz", clock_product="gfz"
        )


class TestGetStationPosition(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processor = GnssRinexProcessor(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch.object(GnssRinexProcessor, "read_observation")
    def test_get_station_position_bogo(self, mock_read_obs):
        # BOGO approximate ECEF position (Borowiec, Poland)
        mock_station = MagicMock()
        mock_station.approx_position = [3655335.0, 1403940.0, 5018005.0]
        mock_read_obs.return_value = mock_station

        pos = self.processor.get_station_position("test.rnx")

        self.assertIn("X", pos)
        self.assertIn("Y", pos)
        self.assertIn("Z", pos)
        self.assertIn("lat", pos)
        self.assertIn("lon", pos)
        self.assertIn("height", pos)

        # BOGO is near 52.2°N, 21.0°E
        self.assertAlmostEqual(pos["lat"], 52.2, delta=0.5)
        self.assertAlmostEqual(pos["lon"], 21.0, delta=0.5)

    @patch.object(GnssRinexProcessor, "read_observation")
    def test_ecef_to_geodetic_equator(self, mock_read_obs):
        # Point on the equator at prime meridian, on the ellipsoid surface
        import numpy as np
        a = 6378137.0  # WGS84 semi-major axis
        mock_station = MagicMock()
        mock_station.approx_position = [a, 0.0, 0.0]
        mock_read_obs.return_value = mock_station

        pos = self.processor.get_station_position("test.rnx")

        self.assertAlmostEqual(pos["lat"], 0.0, places=3)
        self.assertAlmostEqual(pos["lon"], 0.0, places=3)
        self.assertAlmostEqual(pos["height"], 0.0, delta=1.0)


class TestGetObservationSummary(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.processor = GnssRinexProcessor(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch.object(GnssRinexProcessor, "read_observation")
    def test_observation_summary(self, mock_read_obs):
        import pandas as pd

        mock_station = MagicMock()
        mock_station.filename = "BOGO00POL_R_20211000000_01D_30S_MO.crx"
        mock_station.epoch = datetime(2021, 4, 10)
        mock_station.version = 3.04
        mock_station.receiver_type = "LEICA GR25"
        mock_station.antenna_type = "LEIAR25.R4"
        mock_station.interval = 30
        mock_station.observation_types = ["C1C", "L1C", "C2W", "L2W"]
        mock_station.approx_position = [3655335.0, 1403940.0, 5018005.0]

        epochs = pd.to_datetime(["2021-04-10 00:00:00", "2021-04-10 00:00:30"])
        svs = ["G01", "G02", "R01"]
        index = pd.MultiIndex.from_product([epochs, svs], names=["Epoch", "SV"])
        mock_station.observation = pd.DataFrame({
            "C1C": range(6),
            "SYSTEM": ["G", "G", "R", "G", "G", "R"],
        }, index=index)

        mock_read_obs.return_value = mock_station

        summary = self.processor.get_observation_summary("test.rnx")

        self.assertEqual(summary["filename"], "BOGO00POL_R_20211000000_01D_30S_MO.crx")
        self.assertEqual(summary["version"], 3.04)
        self.assertEqual(summary["receiver"], "LEICA GR25")
        self.assertEqual(summary["antenna"], "LEIAR25.R4")
        self.assertEqual(summary["interval_s"], 30)
        self.assertEqual(summary["n_epochs"], 2)
        self.assertEqual(summary["n_satellites"], 3)
        self.assertIn("G", summary["systems"])
        self.assertIn("R", summary["systems"])
        self.assertEqual(summary["observation_types"], ["C1C", "L1C", "C2W", "L2W"])


class TestListRinexFiles(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_list_obs_and_nav(self):
        # Create fake RINEX files
        filenames = [
            "BOGO00POL_R_20211000000_01D_30S_MO.crx.gz",
            "BOGO00POL_R_20211000000_01D_GN.rnx.gz",
            "BOGO00POL_R_20211000000_01D_RN.rnx.gz",
        ]
        for fname in filenames:
            with open(os.path.join(self.tmpdir, fname), "wb") as f:
                f.write(b"fake")

        processor = GnssRinexProcessor(self.tmpdir)
        files = processor.list_rinex_files()

        self.assertEqual(len(files["observation"]), 1)
        self.assertEqual(len(files["navigation"]), 2)

    def test_empty_directory(self):
        processor = GnssRinexProcessor(self.tmpdir)
        files = processor.list_rinex_files()

        self.assertEqual(files["observation"], [])
        self.assertEqual(files["navigation"], [])
        self.assertEqual(files["other"], [])

    def test_non_rinex_files_ignored(self):
        for fname in ["readme.txt", "data.csv", "image.png"]:
            with open(os.path.join(self.tmpdir, fname), "w") as f:
                f.write("not rinex")

        processor = GnssRinexProcessor(self.tmpdir)
        files = processor.list_rinex_files()

        total = len(files["observation"]) + len(files["navigation"]) + len(files["other"])
        self.assertEqual(total, 0)


class TestComputeSppWithOrbits(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.orbit_dir = tempfile.mkdtemp()
        self.processor = GnssRinexProcessor(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        shutil.rmtree(self.orbit_dir)

    @patch("gnsspy.sp3_interp")
    @patch("gnsspy.position.position.gnssDataframe")
    @patch("gnsspy.position.position._observation_picker", return_value=[])
    @patch.object(GnssRinexProcessor, "read_observation")
    def test_compute_spp_with_orbits_empty_obs_list(self, mock_read_obs, mock_picker, mock_gnss_df, mock_sp3):
        import pandas as pd

        mock_station = MagicMock()
        mock_station.epoch = datetime(2021, 4, 10).date()
        mock_station.interval = 30
        mock_station.receiver_clock = 0.0
        mock_station.approx_position = [3655335.0, 1403940.0, 5018005.0]
        mock_read_obs.return_value = mock_station

        orbit_index = pd.MultiIndex.from_tuples(
            [("G01", "2021-04-10 00:00:00")], names=["SV", "Epoch"]
        )
        mock_sp3.return_value = pd.DataFrame({"X": [1.0]}, index=orbit_index)
        mock_gnss_df.return_value = pd.DataFrame()

        result = self.processor.compute_spp_with_orbits(
            "test.rnx", self.orbit_dir, system="G", cut_off=7.0,
        )

        # With empty observation_list (< 2), should return empty DataFrame
        self.assertTrue(result.empty)

    @patch("gnsspy.sp3_interp")
    @patch("gnsspy.position.position.gnssDataframe")
    @patch("gnsspy.position.position._observation_picker", return_value=[])
    @patch.object(GnssRinexProcessor, "read_observation")
    def test_compute_spp_fixes_receiver_clock_string(self, mock_read_obs, mock_picker, mock_gnss_df, mock_sp3):
        import pandas as pd

        mock_station = MagicMock()
        mock_station.epoch = datetime(2021, 4, 10).date()
        mock_station.interval = 30
        mock_station.receiver_clock = "-.000230923645"
        mock_station.approx_position = [3655335.0, 1403940.0, 5018005.0]
        mock_read_obs.return_value = mock_station

        orbit_index = pd.MultiIndex.from_tuples(
            [("G01", "2021-04-10 00:00:00")], names=["SV", "Epoch"]
        )
        mock_sp3.return_value = pd.DataFrame({"X": [1.0]}, index=orbit_index)
        mock_gnss_df.return_value = pd.DataFrame()

        self.processor.compute_spp_with_orbits("test.rnx", self.orbit_dir)

        # Verify receiver_clock was converted to float
        self.assertIsInstance(mock_station.receiver_clock, float)
        self.assertAlmostEqual(mock_station.receiver_clock, -0.000230923645)

    @patch("gnsspy.sp3_interp")
    @patch("gnsspy.position.position.gnssDataframe")
    @patch("gnsspy.position.position._observation_picker", return_value=[])
    @patch.object(GnssRinexProcessor, "read_observation")
    def test_compute_spp_restores_cwd(self, mock_read_obs, mock_picker, mock_gnss_df, mock_sp3):
        import pandas as pd

        mock_station = MagicMock()
        mock_station.epoch = datetime(2021, 4, 10).date()
        mock_station.interval = 30
        mock_station.receiver_clock = 0.0
        mock_station.approx_position = [3655335.0, 1403940.0, 5018005.0]
        mock_read_obs.return_value = mock_station

        orbit_index = pd.MultiIndex.from_tuples(
            [("G01", "2021-04-10 00:00:00")], names=["SV", "Epoch"]
        )
        mock_sp3.return_value = pd.DataFrame({"X": [1.0]}, index=orbit_index)
        mock_gnss_df.return_value = pd.DataFrame()

        original_cwd = os.getcwd()
        self.processor.compute_spp_with_orbits("test.rnx", self.orbit_dir)

        # CWD should be restored even after orbit interpolation
        self.assertEqual(os.getcwd(), original_cwd)

    @patch("gnsspy.sp3_interp", side_effect=Exception("orbit error"))
    @patch.object(GnssRinexProcessor, "read_observation")
    def test_compute_spp_restores_cwd_on_error(self, mock_read_obs, mock_sp3):
        mock_station = MagicMock()
        mock_station.epoch = datetime(2021, 4, 10).date()
        mock_station.interval = 30
        mock_station.receiver_clock = 0.0
        mock_station.approx_position = [3655335.0, 1403940.0, 5018005.0]
        mock_read_obs.return_value = mock_station

        original_cwd = os.getcwd()

        with self.assertRaises(Exception):
            self.processor.compute_spp_with_orbits("test.rnx", self.orbit_dir)

        # CWD must be restored even on error
        self.assertEqual(os.getcwd(), original_cwd)


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS"),
    "Set RUN_INTEGRATION_TESTS=1 to run integration tests with real RINEX data"
)
class TestGnssRinexProcessorIntegration(unittest.TestCase):
    """
    Integration tests using real downloaded RINEX data from
    data/gnss/rinex/2021/bogo_pl/rinex_data/.

    Requires gnsspy and actual RINEX files.

    Run with: RUN_INTEGRATION_TESTS=1 python -m pytest <this_file> -k Integration -v
    """

    DATA_DIR = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", "..", "..",
        "data", "gnss", "rinex", "2021", "bogo_pl", "rinex_data"
    )

    @classmethod
    def setUpClass(cls):
        cls.DATA_DIR = os.path.normpath(cls.DATA_DIR)
        if not os.path.isdir(cls.DATA_DIR):
            raise unittest.SkipTest(f"RINEX data directory not found: {cls.DATA_DIR}")
        cls.processor = GnssRinexProcessor(cls.DATA_DIR)

    def test_list_rinex_files(self):
        files = self.processor.list_rinex_files()
        self.assertGreater(len(files["observation"]), 0, "No observation files found")
        self.assertGreater(len(files["navigation"]), 0, "No navigation files found")

    def test_read_observation(self):
        files = self.processor.list_rinex_files()
        obs_file = files["observation"][0]
        obs = self.processor.read_observation(obs_file)

        self.assertIsNotNone(obs)
        self.assertIsNotNone(obs.observation)
        self.assertGreater(len(obs.observation), 0)
        self.assertGreater(obs.interval, 0)

    def test_read_navigation(self):
        files = self.processor.list_rinex_files()
        nav_file = files["navigation"][0]
        nav = self.processor.read_navigation(nav_file)

        self.assertIsNotNone(nav)
        self.assertGreater(len(nav), 0)

    def test_get_station_position(self):
        files = self.processor.list_rinex_files()
        obs_file = files["observation"][0]
        pos = self.processor.get_station_position(obs_file)

        self.assertIn("lat", pos)
        self.assertIn("lon", pos)
        self.assertIn("height", pos)
        # BOGO station is near 52.2°N, 21.0°E
        self.assertAlmostEqual(pos["lat"], 52.2, delta=1.0)
        self.assertAlmostEqual(pos["lon"], 21.0, delta=1.0)

    def test_get_observation_summary(self):
        files = self.processor.list_rinex_files()
        obs_file = files["observation"][0]
        summary = self.processor.get_observation_summary(obs_file)

        self.assertIn("n_epochs", summary)
        self.assertIn("n_satellites", summary)
        self.assertGreater(summary["n_epochs"], 0)
        self.assertGreater(summary["n_satellites"], 0)

    def test_compute_tropospheric_delay(self):
        files = self.processor.list_rinex_files()
        obs_file = files["observation"][0]
        delay = self.processor.compute_tropospheric_delay(obs_file)

        # Tropospheric zenith delay is typically 2.0 - 2.6 meters
        self.assertGreater(delay, 1.5)
        self.assertLess(delay, 3.0)


if __name__ == "__main__":
    unittest.main()
