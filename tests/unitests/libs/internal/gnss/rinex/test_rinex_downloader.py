import unittest
import os
import gzip
import time
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock

from libs.internal.gnss.rinex.rinex_downloader import (
    RinexDownloader,
    RinexFile,
    parse_rinex_filename,
    date_to_doy,
    _LinkParser,
    NETWORKS,
    BASE_URL,
)


# Realistic BKG directory listing HTML snippet
SAMPLE_DIRECTORY_HTML = """
<html><body>
<h1>Index of /root_ftp/IGS/obs/2020/004/</h1>
<table>
<tr><td><a href="../">../</a></td></tr>
<tr><td><a href="ACOR00ESP_R_20200040000_01D_30S_MO.crx.gz">ACOR00ESP_R_20200040000_01D_30S_MO.crx.gz</a></td></tr>
<tr><td><a href="ACOR00ESP_R_20200040000_01D_GN.rnx.gz">ACOR00ESP_R_20200040000_01D_GN.rnx.gz</a></td></tr>
<tr><td><a href="ACOR00ESP_R_20200040000_01D_EN.rnx.gz">ACOR00ESP_R_20200040000_01D_EN.rnx.gz</a></td></tr>
<tr><td><a href="ACOR00ESP_R_20200040000_01D_RN.rnx.gz">ACOR00ESP_R_20200040000_01D_RN.rnx.gz</a></td></tr>
<tr><td><a href="BRUX00BEL_R_20200040000_01D_30S_MO.crx.gz">BRUX00BEL_R_20200040000_01D_30S_MO.crx.gz</a></td></tr>
<tr><td><a href="BRUX00BEL_R_20200040000_01D_GN.rnx.gz">BRUX00BEL_R_20200040000_01D_GN.rnx.gz</a></td></tr>
<tr><td><a href="WTZR00DEU_R_20200040000_01D_30S_MO.crx.gz">WTZR00DEU_R_20200040000_01D_30S_MO.crx.gz</a></td></tr>
</table>
</body></html>
"""


class TestParseRinexFilename(unittest.TestCase):

    def test_observation_file_with_sampling(self):
        rf = parse_rinex_filename("ABMF00GLP_R_20200040000_01D_30S_MO.crx.gz")
        self.assertIsNotNone(rf)
        self.assertEqual(rf.station, "ABMF")
        self.assertEqual(rf.monument, "00")
        self.assertEqual(rf.country, "GLP")
        self.assertEqual(rf.source, "R")
        self.assertEqual(rf.year, 2020)
        self.assertEqual(rf.doy, 4)
        self.assertEqual(rf.hour, 0)
        self.assertEqual(rf.minute, 0)
        self.assertEqual(rf.period, "01D")
        self.assertEqual(rf.sampling, "30S")
        self.assertEqual(rf.data_type, "MO")
        self.assertTrue(rf.is_observation)
        self.assertFalse(rf.is_navigation)

    def test_observation_file_without_sampling(self):
        rf = parse_rinex_filename("ACOR00ESP_R_20200040000_01D_MO.rnx.gz")
        self.assertIsNotNone(rf)
        self.assertEqual(rf.station, "ACOR")
        self.assertEqual(rf.sampling, "")
        self.assertEqual(rf.data_type, "MO")
        self.assertTrue(rf.is_observation)

    def test_navigation_file(self):
        rf = parse_rinex_filename("ACOR00ESP_R_20200040000_01D_GN.rnx.gz")
        self.assertIsNotNone(rf)
        self.assertEqual(rf.data_type, "GN")
        self.assertTrue(rf.is_navigation)
        self.assertFalse(rf.is_observation)
        self.assertEqual(rf.constellation, "GPS")

    def test_galileo_navigation(self):
        rf = parse_rinex_filename("ACOR00ESP_R_20200040000_01D_EN.rnx.gz")
        self.assertIsNotNone(rf)
        self.assertEqual(rf.constellation, "Galileo")
        self.assertEqual(rf.description, "Galileo Navigation")

    def test_crx_format(self):
        rf = parse_rinex_filename("WTZR00DEU_R_20200040000_01D_MO.crx.gz")
        self.assertIsNotNone(rf)
        self.assertEqual(rf.station, "WTZR")
        self.assertEqual(rf.country, "DEU")
        self.assertTrue(rf.is_observation)

    def test_without_gz_extension(self):
        rf = parse_rinex_filename("ACOR00ESP_R_20200040000_01D_MO.rnx")
        self.assertIsNotNone(rf)
        self.assertEqual(rf.station, "ACOR")

    def test_hourly_file(self):
        rf = parse_rinex_filename("ACOR00ESP_R_20200041200_01H_MO.rnx.gz")
        self.assertIsNotNone(rf)
        self.assertEqual(rf.period, "01H")
        self.assertEqual(rf.hour, 12)
        self.assertEqual(rf.minute, 0)

    def test_stream_source(self):
        rf = parse_rinex_filename("ACOR00ESP_S_20200040000_01D_MO.rnx.gz")
        self.assertIsNotNone(rf)
        self.assertEqual(rf.source, "S")

    def test_invalid_filename_returns_none(self):
        self.assertIsNone(parse_rinex_filename("not_a_rinex_file.txt"))
        self.assertIsNone(parse_rinex_filename(""))
        self.assertIsNone(parse_rinex_filename("ACOR00ESP_R_2020004_01D_MO.rnx.gz"))

    def test_numeric_station_code(self):
        rf = parse_rinex_filename("00NA00USA_R_20200040000_01D_MO.rnx.gz")
        self.assertIsNotNone(rf)
        self.assertEqual(rf.station, "00NA")


class TestRinexFileProperties(unittest.TestCase):

    def setUp(self):
        self.rf = parse_rinex_filename("ACOR00ESP_R_20200040000_01D_MO.rnx.gz")

    def test_date_property(self):
        # DOY 4 of 2020 = January 4, 2020
        expected = datetime(2020, 1, 4, 0, 0)
        self.assertEqual(self.rf.date, expected)

    def test_date_with_time(self):
        rf = parse_rinex_filename("ACOR00ESP_R_20200041430_01H_MO.rnx.gz")
        expected = datetime(2020, 1, 4, 14, 30)
        self.assertEqual(rf.date, expected)

    def test_description_known_type(self):
        self.assertEqual(self.rf.description, "Mixed Observation")

    def test_description_navigation(self):
        rf = parse_rinex_filename("ACOR00ESP_R_20200040000_01D_RN.rnx.gz")
        self.assertEqual(rf.description, "GLONASS Navigation")

    def test_constellation_mapping(self):
        cases = {
            "GO": "GPS", "RO": "GLONASS", "EO": "Galileo",
            "CO": "BDS", "MO": "Mixed",
        }
        for dtype, expected_const in cases.items():
            fname = f"ACOR00ESP_R_20200040000_01D_{dtype}.rnx.gz"
            rf = parse_rinex_filename(fname)
            self.assertEqual(rf.constellation, expected_const, f"Failed for {dtype}")


class TestDateToDoy(unittest.TestCase):

    def test_jan_1(self):
        self.assertEqual(date_to_doy(datetime(2020, 1, 1)), (2020, 1))

    def test_jan_4(self):
        self.assertEqual(date_to_doy(datetime(2020, 1, 4)), (2020, 4))

    def test_leap_year_dec_31(self):
        self.assertEqual(date_to_doy(datetime(2020, 12, 31)), (2020, 366))

    def test_non_leap_year_dec_31(self):
        self.assertEqual(date_to_doy(datetime(2023, 12, 31)), (2023, 365))

    def test_mar_1_leap(self):
        self.assertEqual(date_to_doy(datetime(2020, 3, 1)), (2020, 61))


class TestLinkParser(unittest.TestCase):

    def test_parse_directory_listing(self):
        parser = _LinkParser()
        parser.feed(SAMPLE_DIRECTORY_HTML)
        # 7 RINEX links + 1 parent directory link
        self.assertEqual(len(parser.links), 8)
        self.assertIn("ACOR00ESP_R_20200040000_01D_30S_MO.crx.gz", parser.links)

    def test_empty_html(self):
        parser = _LinkParser()
        parser.feed("<html><body>No links here</body></html>")
        self.assertEqual(len(parser.links), 0)


class TestRinexDownloaderInit(unittest.TestCase):

    def test_valid_network(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = RinexDownloader(output_dir=tmpdir, network="EUREF")
            self.assertEqual(dl.network, "EUREF")
            self.assertEqual(dl.output_dir, tmpdir)

    def test_invalid_network_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                RinexDownloader(output_dir=tmpdir, network="INVALID")

    def test_default_network_is_igs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = RinexDownloader(output_dir=tmpdir)
            self.assertEqual(dl.network, "IGS")

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "sub", "dir")
            dl = RinexDownloader(output_dir=nested)
            self.assertTrue(os.path.isdir(nested))

    def test_build_day_url(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = RinexDownloader(output_dir=tmpdir, network="EUREF")
            url = dl._build_day_url(2020, 4)
            self.assertEqual(url, f"{BASE_URL}/EUREF/obs/2020/004/")


class TestRinexDownloaderListFiles(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.dl = RinexDownloader(output_dir=self.tmpdir, network="EUREF")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _mock_get(self, status_code=200, text=SAMPLE_DIRECTORY_HTML):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.text = text
        mock_resp.raise_for_status = MagicMock()
        if status_code >= 400:
            from requests.exceptions import HTTPError
            mock_resp.raise_for_status.side_effect = HTTPError(response=mock_resp)
        return mock_resp

    @patch("requests.Session.get")
    def test_list_all_files(self, mock_get):
        mock_get.return_value = self._mock_get()
        files = self.dl.list_files(datetime(2020, 1, 4))
        # 7 valid RINEX files in sample HTML
        self.assertEqual(len(files), 7)

    @patch("requests.Session.get")
    def test_list_filter_by_station(self, mock_get):
        mock_get.return_value = self._mock_get()
        files = self.dl.list_files(datetime(2020, 1, 4), station="ACOR")
        # ACOR: 1 obs (30S_MO.crx.gz) + 3 nav (GN, EN, RN)
        self.assertEqual(len(files), 4)
        for rf in files:
            self.assertEqual(rf.station, "ACOR")

    @patch("requests.Session.get")
    def test_list_filter_by_station_case_insensitive(self, mock_get):
        mock_get.return_value = self._mock_get()
        files = self.dl.list_files(datetime(2020, 1, 4), station="acor")
        self.assertEqual(len(files), 4)

    @patch("requests.Session.get")
    def test_list_filter_by_data_type(self, mock_get):
        mock_get.return_value = self._mock_get()
        files = self.dl.list_files(datetime(2020, 1, 4), data_type="MO")
        # ACOR MO, BRUX MO, WTZR MO (crx) = 3
        self.assertEqual(len(files), 3)
        for rf in files:
            self.assertEqual(rf.data_type, "MO")

    @patch("requests.Session.get")
    def test_list_filter_station_and_type(self, mock_get):
        mock_get.return_value = self._mock_get()
        files = self.dl.list_files(datetime(2020, 1, 4), station="ACOR", data_type="MO")
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].station, "ACOR")
        self.assertEqual(files[0].data_type, "MO")

    @patch("requests.Session.get")
    def test_list_no_matching_station(self, mock_get):
        mock_get.return_value = self._mock_get()
        files = self.dl.list_files(datetime(2020, 1, 4), station="ZZZZ")
        self.assertEqual(len(files), 0)

    @patch("requests.Session.get")
    def test_list_404_returns_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        from requests.exceptions import HTTPError
        mock_resp.raise_for_status.side_effect = HTTPError(response=mock_resp)
        mock_get.return_value = mock_resp
        files = self.dl.list_files(datetime(2020, 1, 4))
        self.assertEqual(len(files), 0)

    @patch("requests.Session.get")
    def test_list_connection_error_returns_empty(self, mock_get):
        from requests.exceptions import ConnectionError
        mock_get.side_effect = ConnectionError("Connection refused")
        files = self.dl.list_files(datetime(2020, 1, 4))
        self.assertEqual(len(files), 0)


class TestRinexDownloaderDownloadFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.dl = RinexDownloader(output_dir=self.tmpdir, network="EUREF")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("requests.Session.get")
    def test_download_file_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_content.return_value = [b"fake rinex data"]
        mock_get.return_value = mock_resp

        filename = "ACOR00ESP_R_20200040000_01D_MO.rnx.gz"
        result = self.dl.download_file(datetime(2020, 1, 4), filename)

        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        with open(result, "rb") as f:
            self.assertEqual(f.read(), b"fake rinex data")

    @patch("requests.Session.get")
    def test_download_file_skip_existing(self, mock_get):
        filename = "ACOR00ESP_R_20200040000_01D_MO.rnx.gz"
        existing_path = os.path.join(self.tmpdir, filename)
        with open(existing_path, "wb") as f:
            f.write(b"existing")

        result = self.dl.download_file(datetime(2020, 1, 4), filename, overwrite=False)

        self.assertEqual(result, existing_path)
        mock_get.assert_not_called()

    @patch("requests.Session.get")
    def test_download_file_overwrite_existing(self, mock_get):
        filename = "ACOR00ESP_R_20200040000_01D_MO.rnx.gz"
        existing_path = os.path.join(self.tmpdir, filename)
        with open(existing_path, "wb") as f:
            f.write(b"old data")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.iter_content.return_value = [b"new data"]
        mock_get.return_value = mock_resp

        result = self.dl.download_file(datetime(2020, 1, 4), filename, overwrite=True)

        self.assertIsNotNone(result)
        with open(result, "rb") as f:
            self.assertEqual(f.read(), b"new data")

    @patch("requests.Session.get")
    def test_download_file_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        from requests.exceptions import HTTPError
        mock_resp.raise_for_status.side_effect = HTTPError(response=mock_resp)
        mock_get.return_value = mock_resp

        result = self.dl.download_file(
            datetime(2020, 1, 4), "ZZZZ00XXX_R_20200040000_01D_MO.rnx.gz"
        )
        self.assertIsNone(result)

    @patch("requests.Session.get")
    def test_download_file_connection_error_cleans_partial(self, mock_get):
        from requests.exceptions import ConnectionError

        filename = "ACOR00ESP_R_20200040000_01D_MO.rnx.gz"
        partial_path = os.path.join(self.tmpdir, filename)
        # Simulate a partial file left by a prior attempt
        with open(partial_path, "wb") as f:
            f.write(b"partial")

        mock_get.side_effect = ConnectionError("Connection lost")
        result = self.dl.download_file(datetime(2020, 1, 4), filename, overwrite=True)

        self.assertIsNone(result)
        self.assertFalse(os.path.exists(partial_path))


class TestRinexDownloaderDownloadStation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.dl = RinexDownloader(output_dir=self.tmpdir, network="EUREF")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch.object(RinexDownloader, "download_file")
    @patch.object(RinexDownloader, "list_files")
    def test_download_obs_only(self, mock_list, mock_download):
        mock_list.return_value = [
            parse_rinex_filename("ACOR00ESP_R_20200040000_01D_30S_MO.crx.gz"),
            parse_rinex_filename("ACOR00ESP_R_20200040000_01D_GN.rnx.gz"),
            parse_rinex_filename("ACOR00ESP_R_20200040000_01D_EN.rnx.gz"),
        ]
        mock_download.return_value = "/fake/path.crx.gz"

        paths = self.dl.download_station("ACOR", datetime(2020, 1, 4), obs=True, nav=False)

        # Only MO should be downloaded (1 obs file), nav files excluded
        self.assertEqual(len(paths), 1)
        mock_download.assert_called_once()

    @patch.object(RinexDownloader, "download_file")
    @patch.object(RinexDownloader, "list_files")
    def test_download_nav_only(self, mock_list, mock_download):
        mock_list.return_value = [
            parse_rinex_filename("ACOR00ESP_R_20200040000_01D_30S_MO.crx.gz"),
            parse_rinex_filename("ACOR00ESP_R_20200040000_01D_GN.rnx.gz"),
            parse_rinex_filename("ACOR00ESP_R_20200040000_01D_EN.rnx.gz"),
        ]
        mock_download.return_value = "/fake/path.rnx.gz"

        paths = self.dl.download_station("ACOR", datetime(2020, 1, 4), obs=False, nav=True)

        # GN and EN are nav files
        self.assertEqual(len(paths), 2)

    @patch.object(RinexDownloader, "download_file")
    @patch.object(RinexDownloader, "list_files")
    def test_download_obs_and_nav(self, mock_list, mock_download):
        mock_list.return_value = [
            parse_rinex_filename("ACOR00ESP_R_20200040000_01D_30S_MO.crx.gz"),
            parse_rinex_filename("ACOR00ESP_R_20200040000_01D_GN.rnx.gz"),
        ]
        mock_download.return_value = "/fake/path.rnx.gz"

        paths = self.dl.download_station("ACOR", datetime(2020, 1, 4), obs=True, nav=True)

        self.assertEqual(len(paths), 2)

    @patch.object(RinexDownloader, "download_file")
    @patch.object(RinexDownloader, "list_files")
    def test_constellation_filter(self, mock_list, mock_download):
        mock_list.return_value = [
            parse_rinex_filename("ACOR00ESP_R_20200040000_01D_GN.rnx.gz"),
            parse_rinex_filename("ACOR00ESP_R_20200040000_01D_EN.rnx.gz"),
            parse_rinex_filename("ACOR00ESP_R_20200040000_01D_RN.rnx.gz"),
        ]
        mock_download.return_value = "/fake/path.rnx.gz"

        paths = self.dl.download_station(
            "ACOR", datetime(2020, 1, 4), obs=False, nav=True, constellation="E"
        )

        # Only Galileo nav
        self.assertEqual(len(paths), 1)

    @patch.object(RinexDownloader, "list_files")
    def test_no_files_available(self, mock_list):
        mock_list.return_value = []
        paths = self.dl.download_station("ZZZZ", datetime(2020, 1, 4))
        self.assertEqual(len(paths), 0)


class TestRinexDownloaderDateRange(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.dl = RinexDownloader(output_dir=self.tmpdir, network="IGS")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch.object(RinexDownloader, "download_station")
    def test_date_range_calls_per_day(self, mock_dl_station):
        mock_dl_station.return_value = ["/fake/path.rnx.gz"]

        results = self.dl.download_date_range(
            "ACOR", datetime(2020, 1, 1), datetime(2020, 1, 3)
        )

        self.assertEqual(mock_dl_station.call_count, 3)
        self.assertEqual(len(results), 3)
        self.assertIn("2020-01-01", results)
        self.assertIn("2020-01-02", results)
        self.assertIn("2020-01-03", results)

    @patch.object(RinexDownloader, "download_station")
    def test_date_range_single_day(self, mock_dl_station):
        mock_dl_station.return_value = ["/fake/path.rnx.gz"]

        results = self.dl.download_date_range(
            "ACOR", datetime(2020, 1, 1), datetime(2020, 1, 1)
        )

        self.assertEqual(mock_dl_station.call_count, 1)
        self.assertEqual(len(results), 1)

    @patch.object(RinexDownloader, "download_station")
    def test_date_range_skips_empty_days(self, mock_dl_station):
        # Day 1: has files, Day 2: empty, Day 3: has files
        mock_dl_station.side_effect = [
            ["/fake/day1.rnx.gz"],
            [],
            ["/fake/day3.rnx.gz"],
        ]

        results = self.dl.download_date_range(
            "ACOR", datetime(2020, 1, 1), datetime(2020, 1, 3)
        )

        self.assertEqual(len(results), 2)
        self.assertNotIn("2020-01-02", results)


class TestRinexDownloaderMultiStation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.dl = RinexDownloader(output_dir=self.tmpdir, network="EUREF")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch.object(RinexDownloader, "download_station")
    def test_download_multiple_stations(self, mock_dl_station):
        mock_dl_station.return_value = ["/fake/path.rnx.gz"]

        results = self.dl.download_stations(
            ["ACOR", "BRUX", "WTZR"], datetime(2020, 1, 4)
        )

        self.assertEqual(mock_dl_station.call_count, 3)
        self.assertEqual(len(results), 3)
        self.assertIn("ACOR", results)
        self.assertIn("BRUX", results)
        self.assertIn("WTZR", results)

    @patch.object(RinexDownloader, "download_station")
    def test_multi_station_skips_empty(self, mock_dl_station):
        mock_dl_station.side_effect = [
            ["/fake/acor.rnx.gz"],
            [],  # BRUX has nothing
            ["/fake/wtzr.rnx.gz"],
        ]

        results = self.dl.download_stations(
            ["ACOR", "BRUX", "WTZR"], datetime(2020, 1, 4)
        )

        self.assertEqual(len(results), 2)
        self.assertNotIn("BRUX", results)


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS"),
    "Set RUN_INTEGRATION_TESTS=1 to run real download tests"
)
class TestRinexDownloaderIntegration(unittest.TestCase):
    """
    Integration tests that hit the real BKG server.

    Run with: RUN_INTEGRATION_TESTS=1 python -m pytest <this_file> -k Integration -v
    Uses EUREF/obs/2020/004 (Jan 4 2020) — a well-established archive date.
    File listing is fetched once in setUpClass and shared across all tests
    to avoid rate-limiting from the BKG server.
    """

    NETWORK = "EUREF"
    STATION = "ACOR"
    DATE = datetime(2020, 1, 4)

    _all_files = None       # cached full listing
    _station_files = None   # cached station-filtered listing

    @classmethod
    def setUpClass(cls):
        cls._class_tmpdir = tempfile.mkdtemp(prefix="rinex_integration_")
        dl = RinexDownloader(output_dir=cls._class_tmpdir, network=cls.NETWORK)
        cls._all_files = dl.list_files(cls.DATE)
        time.sleep(1)
        cls._station_files = dl.list_files(cls.DATE, station=cls.STATION)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._class_tmpdir)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="rinex_integration_")
        self.dl = RinexDownloader(output_dir=self.tmpdir, network=self.NETWORK)
        time.sleep(1)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_list_files_returns_results(self):
        files = self._all_files
        self.assertGreater(len(files), 0, "BKG listing returned no files")

        for rf in files:
            self.assertIsInstance(rf, RinexFile)
            self.assertEqual(rf.year, 2020)
            self.assertEqual(rf.doy, 4)

    def test_list_files_station_filter(self):
        files = self._station_files
        self.assertGreater(len(files), 0, f"No files found for station {self.STATION}")
        for rf in files:
            self.assertEqual(rf.station, self.STATION)

    def test_download_single_navigation_file(self):
        nav_files = [rf for rf in self._station_files if rf.is_navigation]
        self.assertGreater(len(nav_files), 0, "No navigation files found for ACOR")

        target = nav_files[0]
        local_path = self.dl.download_file(self.DATE, target.filename)

        self.assertIsNotNone(local_path, f"Download returned None for {target.filename}")
        self.assertTrue(os.path.exists(local_path))

        file_size = os.path.getsize(local_path)
        self.assertGreater(file_size, 0, "Downloaded file is empty")

        # Verify it's a valid gzip file
        with gzip.open(local_path, "rb") as f:
            header = f.read(128)
        self.assertGreater(len(header), 0, "Cannot decompress downloaded .gz file")

    def test_download_observation_and_navigation_files(self):
        # Find a station that has both obs and nav in the full listing
        from collections import defaultdict
        station_types = defaultdict(lambda: {"obs": [], "nav": []})
        for rf in self._all_files:
            if rf.is_observation:
                station_types[rf.station]["obs"].append(rf)
            elif rf.is_navigation:
                station_types[rf.station]["nav"].append(rf)

        target_station = None
        for stn, types in station_types.items():
            if types["obs"] and types["nav"]:
                target_station = stn
                break

        if target_station is None:
            self.skipTest("No station found with both obs and nav files on this date")

        obs_file = station_types[target_station]["obs"][0]
        nav_file = station_types[target_station]["nav"][0]

        paths = []
        for rf in [nav_file, obs_file]:
            path = self.dl.download_file(self.DATE, rf.filename)
            self.assertIsNotNone(path, f"Download failed for {rf.filename}")
            paths.append(path)
            time.sleep(1)

        for p in paths:
            self.assertTrue(os.path.exists(p))
            self.assertGreater(os.path.getsize(p), 0)

        downloaded = [parse_rinex_filename(os.path.basename(p)) for p in paths]
        self.assertTrue(any(rf.is_observation for rf in downloaded))
        self.assertTrue(any(rf.is_navigation for rf in downloaded))

    def test_download_skips_existing_by_default(self):
        nav_files = [rf for rf in self._station_files if rf.is_navigation]
        target = nav_files[0]

        # First download
        path1 = self.dl.download_file(self.DATE, target.filename)
        self.assertIsNotNone(path1)
        mtime1 = os.path.getmtime(path1)

        # Second download — should skip (file untouched, no network call)
        path2 = self.dl.download_file(self.DATE, target.filename, overwrite=False)
        mtime2 = os.path.getmtime(path2)

        self.assertEqual(path1, path2)
        self.assertEqual(mtime1, mtime2, "File was re-downloaded when it should have been skipped")


if __name__ == "__main__":
    unittest.main()
