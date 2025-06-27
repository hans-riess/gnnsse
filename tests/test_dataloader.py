import os
import tempfile
import json
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from src.dataloader import DataLoader

# Minimal test label set
TEST_LABELS = [
    {"siteID": "PORA", "label": 1},
    {"siteID": "TAWA", "label": 0},
    {"siteID": "TURI", "label": 0},
    {"siteID": "WPUK", "label": 1},
    {"siteID": "WPAW", "label": 0}
]

TEST_DATA = [
    {"siteID": "PORA", "t": "2021-01-01T00:00:00Z", "e": 1.0, "n": 2.0, "u": 3.0, "geometry": Point(174.7, -41.3)},
    {"siteID": "TAWA", "t": "2021-01-01T00:00:00Z", "e": 1.1, "n": 2.1, "u": 3.1, "geometry": Point(174.8, -41.2)},
    {"siteID": "TURI", "t": "2021-01-01T00:00:00Z", "e": 1.2, "n": 2.2, "u": 3.2, "geometry": Point(174.9, -41.1)},
    {"siteID": "WPUK", "t": "2021-01-01T00:00:00Z", "e": 1.3, "n": 2.3, "u": 3.3, "geometry": Point(175.0, -41.0)},
    {"siteID": "WPAW", "t": "2021-01-01T00:00:00Z", "e": 1.4, "n": 2.4, "u": 3.4, "geometry": Point(175.1, -40.9)},
    {"siteID": "PORA", "t": "2021-01-02T00:00:00Z", "e": 1.5, "n": 2.5, "u": 3.5, "geometry": Point(174.7, -41.3)},
    {"siteID": "TAWA", "t": "2021-01-02T00:00:00Z", "e": 1.6, "n": 2.6, "u": 3.6, "geometry": Point(174.8, -41.2)},
    {"siteID": "TURI", "t": "2021-01-02T00:00:00Z", "e": 1.7, "n": 2.7, "u": 3.7, "geometry": Point(174.9, -41.1)},
    {"siteID": "WPUK", "t": "2021-01-02T00:00:00Z", "e": 1.8, "n": 2.8, "u": 3.8, "geometry": Point(175.0, -41.0)},
    {"siteID": "WPAW", "t": "2021-01-02T00:00:00Z", "e": 1.9, "n": 2.9, "u": 3.9, "geometry": Point(175.1, -40.9)}
]

def make_test_geojson(path):
    df = pd.DataFrame(TEST_DATA)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")

def make_test_labels(path):
    with open(path, "w") as f:
        json.dump(TEST_LABELS, f)

def test_dataloader_load_and_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        geojson_path = os.path.join(tmpdir, "test_data.geojson")
        labels_path = os.path.join(tmpdir, "test_labels.json")
        make_test_geojson(geojson_path)
        make_test_labels(labels_path)
        # Test DataLoader loads data
        loader = DataLoader(
            data_path=geojson_path,
            label_path=labels_path,
            start_date="2021-01-01T00:00:00Z",
            end_date="2021-01-02T00:00:00Z",
            download=False,
            load=True,
            debug=True
        )
        assert hasattr(loader, "df")
        assert len(loader.df) == 10
        # Test get_graph returns correct shape
        graph = loader.get_graph(k=2)
        assert graph.features[0].shape == (5, 3)  # 5 nodes, 3 features
        assert len(graph.features) == 2  # 2 timestamps
        assert graph.targets[0].shape[0] == 5
        assert graph.positions.shape == (5, 2)
        # Check labels
        assert set(graph.y.tolist()) == {0.0, 1.0} 