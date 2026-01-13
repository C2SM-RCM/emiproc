import warnings

import pytest

from emiproc.exports.geopackage import export_to_geopackage
from emiproc.inventories import Inventory
from emiproc.tests_utils.african_case import african_inv
from emiproc.tests_utils.test_inventories import (
    inv_on_grid_serie2,
    inv_with_pnt_sources,
)


@pytest.fixture(
    params=[african_inv, inv_with_pnt_sources, inv_on_grid_serie2],
    ids=["african_inv", "inv_with_pnt_sources", "inv_on_grid_serie2"],
)
def inv(request):
    return request.param


def test_export_to_geopackage(tmp_path, inv: Inventory):
    import geopandas as gpd
    import fiona

    output_path = tmp_path / f"test_inventory_export_{inv}.gpkg"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        export_to_geopackage(inv, output_path)

        # Verify the file was created
        assert output_path.exists(), "GeoPackage file was not created"

        # Get all layers in the GeoPackage
        layers = fiona.listlayers(output_path)
        assert len(layers) > 0, "No layers found in GeoPackage"

        # Build expected layers list
        expected_layers = []

        # Check for shaped emissions (gdfs)
        if inv.gdfs:
            expected_layers.extend(inv.gdfs.keys())

        # Check for gridded emissions (gdf)
        if hasattr(inv, "gdf") and inv.gdf is not None:
            expected_layers.append("gridded_emissions")

        # Verify all expected layers are present
        assert set(expected_layers) == set(
            layers
        ), f"Layer mismatch. Expected: {set(expected_layers)}, Found: {set(layers)}"

        # Validate each shaped emission layer
        if inv.gdfs:
            for cat, original_gdf in inv.gdfs.items():
                # Read the layer
                layer_gdf = gpd.read_file(output_path, layer=cat)

                # Check data integrity - row count
                assert len(layer_gdf) == len(original_gdf), (
                    f"Layer '{cat}': row count mismatch. "
                    f"Expected {len(original_gdf)}, got {len(layer_gdf)}"
                )

                # Check geometry preservation
                original_null_count = original_gdf.geometry.isna().sum()
                exported_null_count = layer_gdf.geometry.isna().sum()
                assert exported_null_count == original_null_count, (
                    f"Layer '{cat}': geometry null count mismatch. "
                    f"Original: {original_null_count}, Exported: {exported_null_count}"
                )

                # Check that emission columns are present
                for col in original_gdf.columns:
                    if col != "geometry":
                        assert (
                            col in layer_gdf.columns
                        ), f"Layer '{cat}': missing column '{col}'"

                # Verify CRS is preserved
                if original_gdf.crs is not None:
                    assert (
                        layer_gdf.crs == original_gdf.crs
                    ), f"Layer '{cat}': CRS mismatch"

        # Validate gridded emissions layer
        if hasattr(inv, "gdf") and inv.gdf is not None:
            gridded_gdf = gpd.read_file(output_path, layer="gridded_emissions")

            # Check data integrity - row count
            assert len(gridded_gdf) == len(inv.gdf), (
                f"Gridded layer: row count mismatch. "
                f"Expected {len(inv.gdf)}, got {len(gridded_gdf)}"
            )

            # Check geometry preservation
            original_null_count = inv.gdf.geometry.isna().sum()
            exported_null_count = gridded_gdf.geometry.isna().sum()
            assert exported_null_count == original_null_count, (
                f"Gridded layer: geometry null count mismatch. "
                f"Original: {original_null_count}, Exported: {exported_null_count}"
            )

            # Verify CRS is preserved
            if inv.gdf.crs is not None:
                assert gridded_gdf.crs == inv.gdf.crs, "Gridded layer: CRS mismatch"
