import geopandas as gpd
from shapely.geometry import Point

import pyproj
from shapely.ops import transform

def main():
    
    # Read in the geopackage
    constrained_airspace = gpd.read_file('whole_vienna/gis/updated_constrained_airspace.gpkg').to_crs(epsg=4326)

    # get geometry
    poly_geom = list(constrained_airspace.loc[0, 'geometry'].boundary.coords)

    # write to a string
    point_list = [f'{geo_tuple[1]} {geo_tuple[0]}' for geo_tuple in poly_geom]
    
    # combine into one string
    point_string = ' '.join(point_list)
    
    # add poly and name
    const_string = '00:00:00>POLY CONST ' + point_string

    # read in the geopackage
    open_airspace = gpd.read_file('whole_vienna/gis/overall_airspace.gpkg').to_crs(epsg=4326)

    # get geometry
    centroid = list(open_airspace.loc[0, 'geometry'].centroid.coords)
    poly_geom = open_airspace.loc[0, 'geometry']

    # get minimum bounding box around polygon
    box = poly_geom.minimum_rotated_rectangle

    # get coordinates of polygon vertices
    x, y = box.exterior.coords.xy

    wgs84 = pyproj.CRS('EPSG:4326')
    utm = pyproj.CRS('EPSG:32633')
    project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    x = transform(project, x)
    y = transform(project, y)

    print(x,y)

    # get length of bounding box edges
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))

    # get length of polygon as the longest edge of the bounding box
    length = max(edge_length)

    # get width of polygon as the shortest edge of the bounding box
    width = min(edge_length)

    print(width, length)
    # write to a string
    center_point = f'{centroid[0][1]} {centroid[0][0]}'


    # combine into one string
    point_string = ' '.join(point_list)
    
    # add poly and name
    open_string = '00:00:00>CIRCLE OPEN ' + point_string

    # write to file
    with open('airspace.scn', 'w') as f:
        f.write(const_string + '\n')
        f.write(open_string)


if __name__ == '__main__':
    main()

