"""Utilities to use open street map data."""

import logging

import geopandas as gpd
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def osm_json_to_gdf(
    data: dict[str, list[dict[str, any]]],
    extract_tags: list[str] = [],
) -> gpd.GeoDataFrame:
    """Converts an OSM JSON to a GeoDataFrame.

    It proceeds the following way:
    - Nodes are converted to Points
    - Ways are converted to LineStrings or Polygons (if they are closed)
    - Relations are converted to MultiPolygons or GeometryCollections
    - Missing objects (Nodes, Ways, Relations) are ignored
    - objects without tags are ignored (simple geometry probably as ref for other objects)
    - tags are stored as strings in a tag column


    Example usage:

    ```python
    import requests

    url = "https://overpass-api.de/api/interpreter"
    bbox = (8.68, 49.40, 8.72, 49.42)
    query = f"[out:json];nwr({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out;"
    response = requests.get(url, params={"data": query})
    data = response.json()
    gdf = osm_json_to_gdf(data)

    ```

    :param data: OSM JSON data. This can be obtained by making a request to the OSM API.
    :return: A GeoDataFrame with the geometries and tags of the OSM data.

    """

    data_by_id = {x["id"]: x for x in data["elements"]}

    for tag in extract_tags:
        if tag in [
            "id",
            "geometry",
            "name",
            "type",
            "tags",
        ]:
            raise ValueError(f"Cannot extract tag {tag} as it is a reserved name")

    def geometry_from_nodes_ids(node_ids):
        nodes = [
            data_by_id[node_id]
            for node_id in node_ids
            # Remove nodes that are not in the data
            if node_id in data_by_id
        ]
        coords = [(node["lon"], node["lat"]) for node in nodes]

        if node_ids[0] == node_ids[-1]:
            return Polygon(coords)
        return LineString(coords)

    def resolve_way(element):
        """Ways are either polygons or linestrings."""
        return geometry_from_nodes_ids(element["nodes"])

    def resolve_relation(element):

        polys: list[Polygon] = []
        polys_roles: list[str] = []
        incomplete: list[LineString | Point] = []

        current_nodes: list[int] = []
        for member in element["members"]:

            if member["ref"] not in data_by_id:
                logger.debug(f"{member['ref']} not found")
                continue

            if member["type"] == "node":
                node = data_by_id[member["ref"]]
                incomplete.append(Point(node["lon"], node["lat"]))
                continue

            if member["type"] == "relation":
                incomplete.append(resolve_relation(data_by_id[member["ref"]]))
                continue

            assert (
                member["type"] == "way"
            ), f"Unexpected member type: {member['type']} in {element}"

            way = data_by_id[member["ref"]]

            if len(current_nodes) == 0:
                # Start the next polygon
                current_nodes.extend(way["nodes"])
                continue

            # Check if we can add the geom to the current polygon
            if way["nodes"][0] == current_nodes[-1]:
                current_nodes.extend(way["nodes"][1:])
            elif way["nodes"][-1] == current_nodes[0]:
                current_nodes = way["nodes"][:-1] + current_nodes
            elif way["nodes"][-1] == current_nodes[-1]:
                current_nodes.extend(way["nodes"][:-1:-1])
            elif way["nodes"][0] == current_nodes[0]:
                current_nodes = way["nodes"][1::-1] + current_nodes
            else:

                incomplete.append(geometry_from_nodes_ids(current_nodes))
                current_nodes = []
                continue

            # Check if we closed the polygon
            if current_nodes[0] == current_nodes[-1]:
                polys.append(geometry_from_nodes_ids(current_nodes))
                polys_roles.append(member["role"])
                current_nodes = []

        if len(current_nodes) > 0:
            incomplete.append(geometry_from_nodes_ids(current_nodes))

        # TODO: implement that we cut the inside of the polygons
        outer_polys = [p for p, r in zip(polys, polys_roles) if r == "outer"]
        inner_polys = [p for p, r in zip(polys, polys_roles) if r == "inner"]

        # Check if we return a multipolygon or a single polygon
        if len(incomplete) == 0:
            return MultiPolygon(polys)
        else:
            return GeometryCollection(polys + incomplete)

    def get_geometry_osm(element):
        """Resolves the geometry of an element."""
        match element["type"]:
            case "node":
                return Point(element["lon"], element["lat"])
            case "way":
                return resolve_way(element)
            case "relation":
                return resolve_relation(element)
            case _:
                raise ValueError(f"Unknown element type: {element['type']}")

    results_dict = [
        {
            "id": element["id"],
            # "geometry_type": element["tags"]["type"] if "type" in element["tags"] else None,
            "geometry": get_geometry_osm(element),
            "name": element["tags"].get("name", None),
            "type": element["type"],
            "tags": str(element["tags"]),
        }
        | {
            tag: element["tags"].get(tag, None)
            for tag in extract_tags
            if tag in element["tags"]
        }
        for element in data["elements"]
        if "tags" in element
    ]
    results_dict

    return gpd.GeoDataFrame(results_dict, crs="EPSG:4326")
