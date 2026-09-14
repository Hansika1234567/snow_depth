"""
feature_engineering.py
Creates interaction features from terrain and vegetation variables.
"""


def add_engineered_features(df):
    """
    Add three engineered features to the dataset:

    - Elevation_Slope     = elevation * slope
    - Tree_Elevation      = Percent_Tree_Cover * elevation
    - Terrain_Ruggedness  = tri * slope

    These capture non-linear interactions between terrain and vegetation
    variables that are relevant to snow depth estimation.

    Returns the dataframe with the new columns added.
    """
    df = df.copy()
    df["Elevation_Slope"] = df["elevation"] * df["slope"]
    df["Tree_Elevation"] = df["Percent_Tree_Cover"] * df["elevation"]
    df["Terrain_Ruggedness"] = df["tri"] * df["slope"]
    return df
