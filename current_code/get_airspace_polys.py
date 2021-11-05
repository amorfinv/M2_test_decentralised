import geopandas as gpd

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
    const_string = '00:00:00>POLY CONSTRAINED ' + point_string

    # read in the geopackage
    open_airspace = gpd.read_file('whole_vienna/gis/overall_airspace.gpkg').to_crs(epsg=4326)

    # get geometry
    poly_geom = list(open_airspace.loc[0, 'geometry'].boundary.coords)

    # write to a string
    point_list = [f'{geo_tuple[1]} {geo_tuple[0]}' for geo_tuple in poly_geom]
    
    # combine into one string
    point_string = ' '.join(point_list)
    
    # add poly and name
    open_string = '00:00:00>POLY OPEN ' + point_string

    # write to file
    with open('airspace.scn', 'w') as f:
        f.write(const_string + '\n')
        f.write(open_string)


if __name__ == '__main__':
    main()

