import json

def main():
    # Build constrained layer structure
    # Two types of layers in constrained airspace
    layers_0_range, flight_dict, pattern_0 = build_layer_airspace_dict(30, 480, 30, ['C', 'T', 'F'], ['C', 'T', 'F'], 'C', 'range')
    layers_0_levels, _, _ = build_layer_airspace_dict(30, 480, 30, ['C', 'T', 'F'], ['C', 'T', 'F'], 'C', 'levels')

    layers_1_range, _, pattern_1 = build_layer_airspace_dict(30, 480, 30, ['F', 'T', 'C'], ['C', 'T', 'F'], 'C', 'range')
    layers_1_levels, _, _ = build_layer_airspace_dict(30, 480, 30, ['F', 'T', 'C'], ['C', 'T', 'F'], 'C', 'levels')

    # Build open layer structure
    layers_open_range, flight_dict_hdg, pattern_open = build_layer_airspace_dict(30, 480, 30, ['C'], ['C', 'T', 'F'], 'C', 'range')
    layers_open_levels, _, _ = build_layer_airspace_dict(30, 480, 30, ['C'], ['C', 'T', 'F'], 'C', 'levels')
    height_dict, angle_dict, angle_ranges = build_heading_airspace(flight_dict_hdg['levels'], 0, 360, 45)
    layers_open_hdg = {'heights': height_dict, 'angle': angle_dict}
    
    # create constrained layers dictionary
    layers_0 = {'range': layers_0_range, 'levels': layers_0_levels, 'pattern': pattern_0}
    layers_1 = {'range': layers_1_range, 'levels': layers_1_levels, 'pattern': pattern_1}
    layers_open = {'range': layers_open_range, 'levels': layers_open_levels, 'pattern': pattern_open, 'heading': layers_open_hdg}

    # extend the flight_dict to include heading ranges of open airspace
    flight_dict['headings'] = angle_ranges

    # create overall dictionary
    airspace_config = {'height 0': layers_0, 'height 1': layers_1, 'open': layers_open}
    airspace = {'config': airspace_config, 'info': flight_dict}

    # save layers to json
    with open('layers.json', 'w') as fp:
        json.dump(airspace, fp, indent=4)

def build_heading_airspace(flight_levels,  min_angle, max_angle, angle_spacing):
    """ This creates an airsapce layer structure. It has the the height as a key
    and the heading angle range as a value. The heading angle range is a list of two values the lowest and highest angle.
    If there are too many flight levels, then the pattern is repeated.

    Example args:
    flight_levels = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480]
    min_angle = 0
    max_angle = 360
    angle_spacing = 45

    Example output:
    {
        '30': [0, 45],
        '60': [45, 90],
        '90': [90, 135],
        '120': [135, 180],
        '150': [180, 225],
        '180': [225, 270],
        '210': [270, 315],
        '240': [315, 360],
        '270': [0, 45],
        '300': [45, 90],
        '330': [90, 135],
        '360': [135, 180],
        '390': [180, 225],
        '420': [225, 270],
        '450': [270, 315],
        '480': [315, 360]
    }

    Args:
        flight_levels (list): List of flight levels
        min_angle (int): minimum heading angle
        max_angle (int): maximum heading angle
        angle_spacing (int): heading angle spacing

    Returns:
        height_dict (dictionary): layer dictionary where the key is the height and the value is the heading range
        angle_dict (dictionary): layer dictionary where the key is heading range and the value is a list of the heights with the range
        angle_ranges (list): list of heading angles
    """
    # get the angle pattern
    start_ind = min_angle
    end_ind = max_angle + angle_spacing

    # get list of angles
    angles = [idx for idx in range(start_ind, end_ind, angle_spacing)]
    angle_ranges = []

    for idx in range(len(angles) - 1):
        angle_ranges.append(f'{angles[idx]}-{angles[idx + 1]}')

    # check lengths of angle_ranges and flight levels
    len_flight_levels = len(flight_levels)
    len_angle_ranges = len(angle_ranges)

    # edit the angle_ranges list if the number of flight levels is greater than the number of angle ranges
    if len_flight_levels > len_angle_ranges:
        # repeat the angle ranges
        angle_ranges = angle_ranges * (len_flight_levels // len_angle_ranges)
        angle_ranges = angle_ranges + angle_ranges[:len_flight_levels % len_angle_ranges]

    # create dictionary
    height_dict = {}
    for idx, level in enumerate(flight_levels):
        height_dict[level] = f'{angle_ranges[idx]}'
    
    # reverse the dictionary and keep non-unique values
    angle_dict = {v: [k for k, v2 in height_dict.items() if v2 == v] for v in set(height_dict.values())}

    return height_dict, angle_dict, angle_ranges

def build_layer_airspace_dict(min_height, max_height, spacing, pattern, closest_layers, exteme_layer, opt):
    """ This creates an airsapce layer dictionary based on the minimum height, 
    maximum height, spacing, and repeating pattern of the layers. Units can
    be anything but they must be consistent. Minimum and maximum heights are
    heights where aircraft are expected to be. The layer dictionary returned
    contains the layer level/range and information about type of layer and
    levels/ranges of surrounding aircraft. At the moment only works for layers
    that have 3 sub layers and repeat.

    TODO:
        - Accept n_layers as an argument instead of max_height if desired.
        - Accept non-integer values
        - Work with other types. say when there

    Args:
        min_height (int): Minimum flight layer height
        max_height (int): Maximum flight layer height
        spacing (int): Uniform spacing between layers
        pattern (list): List of strings with an identifier of the layer. pattern repeats
        closest_layers (list): Order of layer ids for which to provide information in reeturn dictionary.
        exteme_layer (str): String of desired layer to track the top and bottom layer.
        opt (string): If 'range' return dictionary with keys contaitning range of layer heights.
                      If 'levels' return dictionary with keys containing flight levels.

    Returns:
        [dictionary]: layer dictionary containing the layer level/range as the key (string/int).
                      The value of dictionary is a list. The first entry gives the layer identifier.
                      The next entries contains info about layers surrounding the current layer. The
                      order of this information depends on closest_layers list.
                      
                      Example:
                      layers = build_layer_airspace_dict(25, 500, 25, ['C', 'T', 'F'], ['C', 'T', 'F'], 'C', 'levels')
                      one entry in the dictionary would be as follows:
                      {layer level: [layer id, {bottom 'C' layer level), (top 'C' layer level), (bottom 'T' layer level), 
                                     (top 't' layer), (bottom 'F' layer level), (top 'F' layer level), (lowest extreme layer),
                                     (highest extreme layer)]
                       }
        [dictionary]: layer heights under key='levels', spacing under key='spacing', and pattern under key='pattern'
    """    
    start_ind = min_height
    end_ind = max_height + spacing

    flight_layer_heights = [idx for idx in range(start_ind, end_ind, spacing)]
    n_layers = len(flight_layer_heights)

    n_type_layers = len(pattern)

    layer_dict = {}
    full_layer_pattern = []

    # order of layer info
    order_layer_info = {}
    for id, layer_type in enumerate(closest_layers):
        order_layer_info[layer_type] = id

    counter = 0
    for idx, layer in enumerate(flight_layer_heights):

        idx_layer = counter%n_type_layers
        layer_choice = pattern[idx_layer]

        # build pattern around layer
        layers_below = pattern[:idx_layer]
        layers_above = pattern[idx_layer+1:]

        if not layers_below:
            layers_below = layers_above

        if not layers_above:
            layers_above = layers_below
        
        layer_pattern = layers_below + [layer_choice] + layers_above

        while len(layer_pattern) != n_type_layers*(2) - 1:
            
            # next layer above
            last_layer = layers_above[-1]
            idx_above = pattern.index(last_layer)
            new_idx = (idx_above + 1)%n_type_layers
            new_last_layer = pattern[new_idx]


            # next layer below
            first_layer = layers_below[0]
            idx_below = pattern.index(first_layer)
            new_idx = idx_below - 1
            new_first_layer = pattern[new_idx]

            # combine to create new local layer pattern
            layer_pattern =[new_first_layer] + layer_pattern + [new_last_layer]

        # get index of current layer in layer pattern
        idx_current_layer = layer_pattern.index(layer_choice)

        # add same layer to layer pattern front and back and add 1 to idx_current_layer
        layer_pattern = [layer_choice] + layer_pattern + [layer_choice]
        idx_current_layer += 1

        # replace layer_choice in middle with current layer text
        layer_pattern = layer_pattern[:idx_current_layer] + ['current_layer'] + layer_pattern[idx_current_layer+1:]

        # add info about other layers
        other_layer_info = []
        for other_layer in closest_layers:
            
            # find index of other_layers in layer pattern
            idx_other_layers = [i for i, j in enumerate(layer_pattern) if j == other_layer]

            for idx_other in idx_other_layers:

                # get layer difference between the current layer and the other layer
                step_idx = idx_other - idx_current_layer

                # add this step to get an index to use in flight_layer_heights
                diff_idx = idx + step_idx

                # If statement catches when we try and get lower or upper layers that don't exist
                if diff_idx < 0 or diff_idx > n_layers - 1:
                    height_other_layer = ''
                else:
                    height_other_layer = layer_output_choice(flight_layer_heights[diff_idx], spacing, opt)
        
                other_layer_info.append(height_other_layer)

        layer_dict_key = layer_output_choice(layer, spacing, opt)
        layer_dict_value = [layer_choice] + other_layer_info
        layer_dict[layer_dict_key] = layer_dict_value

        # create full layer pattern
        full_layer_pattern.append(layer_choice)

        counter += 1

    # get info about desired extreme layers
    exteme_layer_bottom_idx = full_layer_pattern.index(exteme_layer)
    exteme_layer_top_idx = len(full_layer_pattern) - full_layer_pattern[::-1].index(exteme_layer) - 1

    exteme_layer_bottom_height = flight_layer_heights[exteme_layer_bottom_idx]
    exteme_layer_bottom_height = layer_output_choice(exteme_layer_bottom_height, spacing, opt)
    exteme_layer_top_height = flight_layer_heights[exteme_layer_top_idx]
    exteme_layer_top_height = layer_output_choice(exteme_layer_top_height, spacing, opt)

    # add this info to each list inside layer_dict
    for _, value in layer_dict.items():
        value.append(exteme_layer_bottom_height)
        value.append(exteme_layer_top_height)

    # other dictionary
    gen_dict = {'levels': flight_layer_heights, 'spacing': spacing}

    return layer_dict, gen_dict, full_layer_pattern

def layer_output_choice(layer, spacing, opt):
    if opt == 'range':
        layer_dict_key = f'{layer - spacing/2}-{layer + spacing/2}'
    elif opt == 'levels':
        layer_dict_key = layer
    
    return layer_dict_key

if __name__ == '__main__':
    main()
