import json

from numpy.lib.function_base import append

def main():
    # Build layer structure

    # at the moment there are two types of layers
    layers_1_range, flight_dict = build_layer_airspace_dict(25, 500, 25, ['C', 'T', 'F'], ['C', 'T', 'F'], 'range')
    layers_1_levels, _ = build_layer_airspace_dict(25, 500, 25, ['C', 'T', 'F'], ['C', 'T', 'F'], 'levels')

    layers_2_range, _ = build_layer_airspace_dict(25, 500, 25, ['F', 'T', 'C'], ['C', 'T', 'F'], 'range')
    layers_2_levels, _ = build_layer_airspace_dict(25, 500, 25, ['F', 'T', 'C'], ['C', 'T', 'F'], 'levels')

    # create layers dictionary
    layers_1 = {'range': layers_1_range, 'levels': layers_1_levels}
    layers_2 = {'range': layers_2_range, 'levels': layers_2_levels}

    # create overall dictionary
    airspace_config = {'height 1': layers_1, 'height 2': layers_2}
    airspace = {'config': airspace_config, 'info': flight_dict}

    # save layers to json
    with open('layers.json', 'w') as fp:
        json.dump(airspace, fp)

def build_layer_airspace_dict(min_height, max_height, spacing, pattern, closest_layers, opt):
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
        opt (string): If 'range' return dictionary with keys contaitning range of layer heights.
                      If 'levels' return dictionary with keys containing flight levels.

    Returns:
        [dictionary]: layer dictionary containing the layer level/range as the key (string/int).
                      The value of dictionary is a list. The first entry gives the layer identifier.
                      The next entries contains info about layers surrounding the current layer. The
                      order of this information depends on closest_layers list.
                      
                      Example:
                      layers = build_layer_airspace_dict(25, 500, 25, ['C', 'T', 'F'], ['C', 'T', 'F'], 'levels')
                      one entry in the dictionary would be as follows:
                      {layer level: [layer id, {bottom 'C' layer level), (top 'C' layer level), (bottom 'T' layer level), 
                                     (top 't' layer), (bottom 'F' layer level), (top 'F' layer level) 
                                     ]
                       }
        [list]: layer heights
    """    
    start_ind = min_height
    end_ind = max_height + spacing

    flight_layer_heights = [idx for idx in range(start_ind, end_ind, spacing)]
    n_layers = len(flight_layer_heights)

    n_type_layers = len(pattern)

    layer_dict = {}

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
        counter += 1
    
    # other dictionary
    gen_dict = {'levels': flight_layer_heights, 'spacing': spacing}

    return layer_dict, gen_dict

def layer_output_choice(layer, spacing, opt):
    if opt == 'range':
        layer_dict_key = f'{layer - spacing/2}-{layer + spacing/2}'
    elif opt == 'levels':
        layer_dict_key = layer
    
    return layer_dict_key

if __name__ == '__main__':
    main()
