#last updated: 24/06/2025

from ssegnn.graph import StaticGraphTemporalSignal
import pandas as pd
import requests
from tqdm import tqdm
import json
import torch
import numpy as np

class DataLoader(object):
    '''
    label_path: .json file with the linear / non-linear labels from Truttmann et al. 2024
    data_path: .geojson file with the data
    start_date: start date of the data
    end_date: end date of the data
    download: whether to (re)download the data from the Tilde API
    from_file: or not whether to load the data file
    '''
    def __init__(self,
                 data_path,
                 label_path,
                 start_date=pd.Timestamp('2014-01-01 11:59:00+0000', tz='UTC'),
                 end_date=pd.Timestamp('2024-12-31 11:59:00+0000', tz='UTC'),
                 download=False,
                 from_file = False,
                 debug = False,
                 verbose = False,
                ):
        self.start_date = pd.Timestamp(start_date, tz='UTC')
        self.end_date = pd.Timestamp(end_date, tz='UTC')
        self.label_path = label_path
        self.data_path = data_path
        self.debug = debug
        self.verbose = verbose
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)
        if download:
            self.download()
            self.load_data()
        elif from_file:
            self.load_data()
        else:
            raise ValueError("Either download or from_file must be True")
        
        # Debug: Check for missing siteIDs in data
        if self.debug and hasattr(self, 'df'):
            label_siteIDs = set(entry['siteID'] for entry in self.labels)
            data_siteIDs = set(self.df['siteID'].unique())
            missing = label_siteIDs - data_siteIDs
            if missing:
                print(f"[DEBUG] Warning: {len(missing)} siteIDs in labels are missing from the data file:")
                for site in sorted(missing):
                    print(f"  - {site}")
            else:
                print("[DEBUG] All siteIDs in labels are present in the data file.")

    def download(self):
        import geopandas as gpd
        from shapely.geometry import Point

        base_url = 'https://tilde.geonet.org.nz'
        endpoint = 'v4/data/gnss'
        start_date_str = self.start_date.strftime('%Y-%m-%d')
        end_date_str = self.end_date.strftime('%Y-%m-%d')

        siteIDs = [entry['siteID'] for entry in self.labels]

        records = []
        if self.verbose:
            print('Downloading data...')
            print(f'Number of sites: {len(siteIDs)}')
            print(f'Start date: {start_date_str}')
            print(f'End date: {end_date_str}')
            print('\n')

        for siteID in tqdm(siteIDs, disable=not self.verbose):
            try:
                # Fetch east, north, up
                data = {}
                for comp, col, err_col in [
                    ('east', 'e', 'e_err'),
                    ('north', 'n', 'n_err'),
                    ('up', 'u', 'u_err')
                ]:
                    url = f"{base_url}/{endpoint}/{siteID}/displacement/nil/1d/{comp}/{start_date_str}/{end_date_str}"
                    resp = requests.get(url)
                    if resp.status_code == 200:
                        d = resp.json()
                        if isinstance(d, list) and len(d) > 0 and 'data' in d[0]:
                            data[comp] = d[0]
                        else:
                            data[comp] = None
                    else:
                        data[comp] = None

                # Only proceed if all three components are present
                if all(data[c] is not None for c in ['east', 'north', 'up']):
                    # Use east as the reference for timestamps
                    east_data = data['east']['data']
                    lat = data['east'].get('latitude')
                    lon = data['east'].get('longitude')
                    # Build a dict of timestamp to value for each component
                    n_dict = {x['ts']: x for x in data['north']['data']}
                    u_dict = {x['ts']: x for x in data['up']['data']}
                    for e_entry in east_data:
                        ts = e_entry['ts']
                        n_entry = n_dict.get(ts)
                        u_entry = u_dict.get(ts)
                        if n_entry and u_entry:
                            records.append({
                                'siteID': siteID,
                                't': ts,
                                'e': e_entry['val'],
                                'n': n_entry['val'],
                                'u': u_entry['val'],
                                'geometry': Point(lon, lat)
                            })
            except Exception as e:
                print(f"Error fetching data for site {siteID}: {e}")
                continue
        if self.verbose:
            print('Saving data to file...')

        gdf = gpd.GeoDataFrame(records, geometry='geometry', crs="EPSG:4326")
        gdf.to_file(self.data_path, driver="GeoJSON")
        self.df = gdf

        if self.verbose:
            print('Done!')

    def load_data(self):
        '''
        Load the data from the .geojson files
        returns:
            df: pandas dataframe with the data
            gdf_location: geopandas dataframe with the site locations
        '''

        if self.verbose:
            print('Loading data from file...')

        import geopandas as gpd
        self.df = gpd.read_file(self.data_path,driver='GeoJSON')

        if self.verbose:
            print('Done!')
            print('\n')

    def get_graph(self, k=None, r=None):
        '''
        Build a temporal graph dataset from the dataframe.
        Each node is a site, each time step is a measurement timestamp.
        Features are [e, n, u] (or normalized), targets are the site labels.
        '''
        import torch
        from torch_geometric.data import Data
        import torch_geometric.transforms as T
        import numpy as np

        if self.verbose:
            if k is not None:
                print(f'Building k-NN graph with k={k}...')
            elif r is not None:
                print(f'Building radius graph with r={int(r/1000)} km...')
            else:
                raise ValueError("Either k or r must be provided")

        # 1. Get unique siteIDs and their positions
        site_meta = self.df.drop_duplicates('siteID')[['siteID', 'geometry']]
        site_meta['longitude'] = site_meta['geometry'].x
        site_meta['latitude'] = site_meta['geometry'].y
        site_meta = site_meta.sort_values('siteID').reset_index(drop=True)
        siteIDs = site_meta['siteID'].tolist()
        positions = torch.tensor(site_meta[['longitude', 'latitude']].values, dtype=torch.float)

        # 2. Build the graph structure (kNN or radius)
        points = Data(pos=positions)
        if k is not None:
            transform = T.KNNGraph(k=k, force_undirected=True, loop=False)
        elif r is not None:
            transform = T.RadiusGraph(r=r, loop=False)
        else:
            raise ValueError("Either k or r must be provided")
        graph = transform(points)

        # 3. Prepare features for each timestamp
        features = []
        timestamps = sorted(self.df['t'].unique())
        for t in timestamps:
            group = self.df[self.df['t'] == t].set_index('siteID').reindex(siteIDs)
            x = np.stack([group['e'].values, group['n'].values, group['u'].values], axis=1)
            # Always fill NaNs
            if np.isnan(x).any():
                if self.debug:
                    print(f"[DEBUG] NaNs detected and filled in features at timestamp {t}.")
                # Forward fill along sites
                for feat in range(x.shape[1]):
                    mask = np.isnan(x[:, feat])
                    for i in range(1, x.shape[0]):
                        if mask[i] and not mask[i-1]:
                            x[i, feat] = x[i-1, feat]
                    # Backward fill
                    mask = np.isnan(x[:, feat])
                    for i in range(x.shape[0]-2, -1, -1):
                        if mask[i] and not mask[i+1]:
                            x[i, feat] = x[i+1, feat]
                # Fill any remaining NaNs with zero
                x = np.nan_to_num(x, nan=0.0)
            features.append(x)

        # 4. Prepare labels as targets (same for all time steps)
        to_label = {entry['siteID']: entry['label'] for entry in self.labels}
        labels = [to_label.get(siteID, 0) for siteID in siteIDs]  # default to 0 if not found
        labels = torch.tensor(labels,dtype=torch.double)
        targets = [labels for _ in timestamps]

        graph =StaticGraphTemporalSignal(
            edge_index=graph.edge_index,
            edge_weight=graph.edge_attr,
            features=features,
            targets=targets,
            positions=positions
        )

        if self.verbose:
            print('Done!')
            print('\n')
            print(f"Number of nodes: {graph.num_nodes}")
            print(f'Number of node features: {graph.num_node_features}')
            print(f'Number of snapshots: {graph.snapshot_count}')
            print(f'Number of edges: {graph.edge_index.shape[-1]}')
            print('\n')
        return graph