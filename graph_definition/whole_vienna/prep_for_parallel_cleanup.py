from platform import node
import osmnx as ox
import geopandas as gpd
import networkx as nx
from shapely.geometry.point import Point
import graph_funcs
from os import path
from shapely import ops
import pandas as pd
# use osmnx environment here

'''
'''

def main():

    # working path
    gis_data_path = 'gis'

    # Load MultiDigraph from cleanup_graph_2.py
    G = ox.load_graphml(filepath=path.join(gis_data_path, 'streets', 'regrouping_3.graphml'))

    '''--------------------------------------ZONE A----------------------------------'''
    # # TODO:
    ###### cleanup 1

    # edges to delete cleanup 1
    del_edges = [(1476901999, 899064584, 0), (1476901999, 899064584, 0), (1198289029, 1198288976, 0),
                (1198288976, 2425798097, 0), (2425798097, 103646708, 0), (1198289040, 46848761, 0),
                (2424180780, 1404275182, 0), (1404275182, 2424180786, 0), (66694912, 83517944, 0),
                (83517944, 83517968, 0), (83517968, 17322844, 0), (274590512, 274590509, 0), (5368973430, 5368973445, 0)]
    
    G.remove_edges_from(del_edges)

    # nodes to delete
    del_nodes = [33236701, 2424180696, 2522138206, 2522138204, 2424180712, 2424180729, 2522127406,
                2424193904, 2424180761, 2424180774, 2424180780, 2424180790, 2424180796, 2424180800, 
                2424180786, 5368973430]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # # extend edges
    edges_to_extend = [(33208151, 1198289029, 0), (103646708, 46848701, 0), (79704305, 1404275182, 0),
                        (69231121, 66694912, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    # join existing nodes
    new_edge = (199624, 5368973445, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='799'))

    # join existing nodes
    new_edge = (274590530, 199624, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='901'))

    G = ox.graph_from_gdfs(nodes, edges)

    ###### cleanup 2
    # TODO: extend nodes 16614544 and 33238737? Add's connectivity?
    # Also edge 33208011-31673339 and 9921179-9697698
    # edges to delete cleanup 1
    del_edges = [(33238688, 33238717, 0), (33238692, 388103065, 0), (33238694, 33238692, 0), (16614544, 33238694, 0),
                (33238737, 33238700, 0), (33238700, 33238706, 0), (33238706, 33238710, 0), (30486722, 277973070, 0)]
    
    G.remove_edges_from(del_edges)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    G = ox.graph_from_gdfs(nodes, edges)

    ###### cleanup 3
    # TODO:
    # edges to delete cleanup 1
    del_edges = [(33079244, 25276591, 0), (25276591, 25276592, 0), (245498398, 245498397, 0),
                (245498401, 245498398, 0), (24966910, 30696003, 0), (105713654, 105708754, 0),
                (105709707, 105713654, 0)]
    
    G.remove_edges_from(del_edges)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # # extend edges
    edges_to_extend = [(25276591, 25276589, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)


    G = ox.graph_from_gdfs(nodes, edges)

    '''--------------------------------------ZONE B----------------------------------'''

    ###### cleanup 1
    # TODO:
    # edges to delete cleanup 
    del_edges = [(199684, 24950487, 0), (24950487, 24950483, 0), (199638, 199684, 0), (1371105029, 26970606, 0)]
    G.remove_edges_from(del_edges)

    # # nodes to delete
    del_nodes = [69240143, 60842028, 333305349, 24950487, 33344229, 33344222]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    G = ox.graph_from_gdfs(nodes, edges)

    ###### cleanup 2
    # TODO:
    # edges to delete cleanup 
    del_edges = [(655052, 2397733665, 0), (2397522191, 655052, 0), (28982576, 2397522191, 0),
                (28982574, 28982576, 0), (33301301, 28982574, 0), (264123183, 33301301, 0),
                (264123183, 33301301, 0), (28982572, 264123183, 0), (28982571, 28982572, 0),
                (28982570, 28982571, 0), (28982569, 28982570, 0), (1381612052, 28982569, 0),
                (1381612048, 1381612052, 0), (1381612043, 1381612048, 0), (2521182222, 1381612043, 0),
                (2382977342, 2521182222, 0), (2507395450, 2382977342, 0), (2507395454, 2507395450, 0),
                (2507395459, 2507395454, 0), (378721, 2507395459, 0), (75605497, 378721, 0),
                (33182055, 75605497, 0), (33182054, 33182055, 0), (2509699390, 33182054, 0),
                (2457966745, 2509699390, 0)]
    G.remove_edges_from(del_edges)

    # # nodes to delete
    del_nodes = [28982576]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # # extend edges
    edges_to_extend = [(33302038, 28982574, 0), (28982570, 33301995, 0), (33344335, 1381612048, 0),
                        (1381612043, 33344334, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    # new edge with osmid
    osmid = 2382977342
    new_point = Point(16.353314663715775, 48.222038095558695)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '149')

    G = ox.graph_from_gdfs(nodes, edges)

    ###### cleanup 3
    del_edges = [(3358490116, 25778306, 0), (25778306, 2521182222, 0), (3468078361, 3358490116, 0),
                (3468078371, 3468078361, 0), (3358490117, 3468078371, 0), (3358490130, 3358490117, 0),
                (3358490132, 3358490130, 0), (716863740, 3358490132, 0), (33344348, 716863740, 0)]
    G.remove_edges_from(del_edges)
    
    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)
    
    # # extend edges
    edges_to_extend = [(3468078361, 33344381, 0), (33344382, 3468078371, 0), (33344354, 3358490130, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)
    
    G = ox.graph_from_gdfs(nodes, edges)

    ###### cleanup 4
    del_edges = [(242132118, 260604177, 0), (260604177, 260604298, 0), (1114680102, 123854325, 0), 
                 (3704365814, 1114680102, 0), (1114680094, 3704365814, 0)]
    G.remove_edges_from(del_edges)
    
    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)
    
    # join existing nodes
    new_edge = (1114680102, 1114680094, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='60'))

    # new edge with osmid
    osmid = 3704365814
    new_point = Point(16.345352603558577, 48.212467198034425)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '60')

    # # extend edges
    edges_to_extend = [(25267613, 260604177, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)
    
    G = ox.graph_from_gdfs(nodes, edges)

    ###### cleanup 5
    del_edges = [(260604298, 670005, 0), (670005, 260604319, 0), (260604319, 260604332, 0), (260604332, 277838150, 0),
                 (277838150, 260604333, 0), (260604333, 260604392, 0), (260604392, 260604463, 0),
                 (260604463, 260604466, 0), (260604466, 260604690, 0), (17312845, 270929766, 0),
                 (260604690, 270929766, 0), (289468565, 289468505, 0), (277838150, 33183689, 0), (33183693, 199735, 0),
                 (33183696, 33183693, 0), (1604260624, 33183696, 0), (83555096, 1604260624), (256665968, 33183678, 0),
                 (83555096, 2147779384, 0)]
    G.remove_edges_from(del_edges)
    
    # delete nodes
    del_nodes = [33183693, 277838150, 60565719, 1604260624]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)
    
    # join existing nodes
    new_edge = (33183678, 33183679, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='107'))

    # join existing nodes
    new_edge = (33183679, 2147779389, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='88'))

    # # extend edges
    edges_to_extend = [(670006, 670005, 0), (260604319, 17312845, 0), (260604690, 48097392, 0),
                        (2147779384, 2147779389, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)
    
    G = ox.graph_from_gdfs(nodes, edges)

    ###### cleanup 6
    del_edges = [(1363563553, 48753749, 0), (298719753, 1363563553, 0), (30677596, 298719753, 0),
                (48753745, 48753747, 0), (46921615, 46921780, 0), (34591541, 48071926, 0)]
    G.remove_edges_from(del_edges)
    
    # delete nodes
    del_nodes = [298719753, 48753747, 48210081, 48210073, 48071396]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # new edge with osmid
    osmid = 48072011
    new_point = Point(16.37359232685237, 48.20001694580752)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '442')

    # new edge with osmid
    osmid = 34591545
    new_point = Point(16.369738229004405, 48.200880573837885)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '153')

    # new edge with osmid
    osmid = 34591541
    new_point = Point(16.36731150248127, 48.197594972143065)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '452')

    # join existing nodes
    new_edge = (1636160845, 48071926, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='423'))

    # # extend edges
    edges_to_extend = [(1833706351, 1363563553, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)
    
    G = ox.graph_from_gdfs(nodes, edges)

    '''--------------------------------------ZONE C----------------------------------'''
    # TODO: FIX NODE 27377268 and connectivity around 199633
    ###### cleanup 1

    # edges to delete cleanup 1
    del_edges = [(30882785, 1382306305, 0), (30882787, 30882785, 0), (30882776, 30882787, 0),
                (93279698, 93279707, 0), (3703776564, 93279698, 0), (1362664821, 3703776564, 0),
                (341477461, 1362664821, 0), (30685741, 341477461), (77506774, 30685741, 0), (30685754, 77506774, 0),
                (30685739, 30685754, 0), (30685736, 30685739, 0), (30685749, 30685736, 0), (30685748, 30685749, 0), 
                (5919412735, 30685748, 0), (30685749, 30685743, 0), (27375757, 93279698, 0)]
    
    G.remove_edges_from(del_edges)

    # nodes to delete
    del_nodes = [30685749]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    new_edge = (27375731, 30882785, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1360'))

    # extend edges
    edges_to_extend = [(294369999, 30882776, 0), (83686174, 93279707, 0), (249199619, 93279698, 0),
                       (3703776564, 77506145, 0), (341477461, 249202894, 0), (17322862, 30685741, 0),
                       (77506771, 77506774, 0), (30685754, 30692823, 0), (47046953, 30685739, 0),
                       (47046949, 30685736, 0), (30685748, 47046938, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    G = ox.graph_from_gdfs(nodes, edges)


    ###### cleanup 2
    # edges to delete cleanup 2
    del_edges = [(27377268, 2526546837, 0), (2526546837, 144608149, 0), (144608149, 62592209, 0),
                        (62592209, 62598141, 0), (62598141, 60584137, 0), (60584137, 59640976, 0),
                        (59640976, 53169050, 0), (53169050, 60584155, 0), (60584155, 60584168, 0),
                        (60584168, 59640975, 0), (59640975, 78431390, 0), (78431390, 27377267, 0), 
                        (27377267, 86002346, 0), (86002346, 8790237562, 0)]
    
    G.remove_edges_from(del_edges)

    # nodes to delete
    del_nodes = [2526546837]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # extend edges
    edges_to_extend = [(144608149, 144608561, 0), (62598139, 62598141, 0), (60584137, 2309659095, 0),
                       (53169050, 60569923, 0), (213451282, 60584155, 0), (213451284, 60584168, 0),
                       (62598523, 59640975, 0), (78431381, 78431390, 0), (301933391, 27377267, 0),
                       (2309659180, 86002346, 0)]

    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    G = ox.graph_from_gdfs(nodes, edges)
 
    ###### cleanup 3a
    # edges to delete
    # TODO: merge 1185 and 813
    del_edges = [(3249726549, 3249726527, 0), (192000050, 3080800578, 0), (306657000, 306658147, 0),
                (6739899006, 1521925366, 0), (1521925366, 306657000, 0), (86057499, 306657320, 0),
                (6739899006, 86057502, 0), (86057502, 6739898989, 0), (295414245, 61831870, 0),
                (295414234, 295414245, 0), (295414216, 295414234, 0), (199665, 6853942832, 0)]
    G.remove_edges_from(del_edges)

    # nodes to delete 
    del_nodes = [6739847856, 199632, 115450165, 306657000, 86057502]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # join existing nodes
    new_edge = (3249726549, 3249726527, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1165'))

    # join existing nodes
    new_edge = (6739899006, 6739898989, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1185'))

    # extend edges
    edges_to_extend = [(306658147, 306657060, 0), (295414245, 103656328, 0), (103656325, 295414234, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    # new edge with osmid
    osmid = 306657320
    new_point = Point(16.394792537909574, 48.175204960544534)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '1166')

    # new edge with osmid
    osmid = 1521925366
    new_point = Point(16.398295671447663, 48.17501003653269)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '1195')

    # join existing nodes
    new_edge = (199665, 6853942832, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1165'))

    G = ox.graph_from_gdfs(nodes, edges)

    '''--------------------------------------ZONE F----------------------------------'''
    ###### cleanup 1
    # TODO: merge at 32666487
    # edges to delete cleanup 1
    del_edges = [(1816733958, 60571016, 0), (291697212, 33344364, 0), (32666477, 32666495, 0), (309204901, 32666477, 0),
                (59987812, 3575740883, 0), (3534786122, 1965401197, 0)]

    G.remove_edges_from(del_edges)

    # nodes to delete
    del_nodes = [48753702, 60571021, 3514012359, 60571017, 32666477]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # join existing nodes
    new_edge = (48210016, 48210036, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1429'))

    # join existing nodes
    new_edge = (277838181, 60571016, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='74'))

    # join existing nodes
    new_edge = (1816733958, 60571016, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='11'))

    # new edge with osmid
    osmid = 32666487
    new_point = Point(16.37335291206154, 48.21049341095883)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '142')

    G = ox.graph_from_gdfs(nodes, edges)

    '''--------------------------------------ZONE D----------------------------------'''
    ###### cleanup 1

    # edges to delete cleanup 1
    del_edges = [(394906, 2050588871, 0), (2050588871, 48071377, 0), (48071377, 28150496, 0),
                (48071377, 48071367, 0), (48753675, 2050588871, 0), (28150496, 311045197, 0),
                (48753809, 48753805, 0), (291337467, 291337314, 0), (295093595, 48753672, 0),
                (123419800, 60253446, 0)]

    G.remove_edges_from(del_edges)

    # nodes to delete
    del_nodes = [2050588871, 28150496, 48071377, 291337314]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # extend edges
    edges_to_extend = [(48753672, 48753675, 0), (48753805, 48753788, 0), (47998088, 291337467, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    # new edge with osmid
    osmid = 48753672
    new_point = Point(16.377222364044787, 48.20074705805564)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '1195')

    # join existing nodes
    new_edge = (47998088, 60253446, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1406'))
    
    # join existing nodes
    new_edge = (123419800, 60253446, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1408'))

    # join existing nodes
    new_edge = (48071474, 48071367, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='26'))

    G = ox.graph_from_gdfs(nodes, edges)

    ###### cleanup 2

    # edges to delete cleanup 2
    del_edges = [(47998151, 47997793, 0), (119339791, 320095461, 0), (119342046, 119339791, 0),
    (78491486, 78491485, 0), (78491485, 78491483, 0), (75569360, 64850682, 0), (1168873425, 75569360, 0),
    (123430807, 123610502, 0), (123610502, 3227263749, 0)]

    G.remove_edges_from(del_edges)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # join existing nodes
    new_edge = (119339791, 29667406, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1175'))

    # join existing nodes
    new_edge = (119339791, 47998209, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1175'))

    # join existing nodes
    new_edge = (1168873425, 75569360, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1153'))

    # extend edges
    edges_to_extend = [(123610504, 123610502, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    G = ox.graph_from_gdfs(nodes, edges)

    ###### cleanup 3

    # edges to delete cleanup 3
    del_edges = [(60121781, 75580282, 0), (75580282, 60484722, 0), (277120400, 90369223, 0)]

    G.remove_edges_from(del_edges)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # join existing nodes
    new_edge = (60121781, 82721803, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1222'))

    # join existing nodes
    new_edge = (75580282, 1834749079, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1431'))

    # join existing nodes
    new_edge = (60484722, 123623206, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='1442'))

    G = ox.graph_from_gdfs(nodes, edges)


    '''--------------------------------------ZONE E----------------------------------'''
    ###### cleanup 1

    # edges to delete cleanup 1
    del_edges = [(59984193, 293269203, 0), (59744701, 59984193, 0), (1271111971, 59744701, 0),
                (252622032, 1271111971, 0), (1697704037, 554187644, 0), (314246316, 2389850721, 0),
                (1735751453, 1184383510, 0), (1735751453, 1184383510, 0), (497238340, 321939087, 0)]

    G.remove_edges_from(del_edges)

    # remove nodes
    del_nodes = [1271111971, 252622032]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # new edge with osmid
    osmid = 321939087
    new_point = Point(16.39030029187754, 48.21375127463434)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '1493')

    # extend edges
    edges_to_extend = [(59744701, 59839001, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    G = ox.graph_from_gdfs(nodes, edges)

    ###### cleanup 2

    # edges to delete cleanup 2
    del_edges = [(59838945, 2106523975, 0), (684994, 293040456, 0), (513404700, 684994, 0),
                (266655464, 103868511, 0), (60474276, 254314477, 0), (353280664, 60637199, 0),
                (60637196, 353278793, 0)]

    G.remove_edges_from(del_edges)

    # # remove nodes
    del_nodes = [293040450, 556664271, 556664273, 60637197, 353278793, 60637196]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # new edge with osmid
    osmid = 266655464
    new_point = Point(16.39172421790962, 48.21730462887545)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '51')

    # extend edges
    edges_to_extend = [(293040453, 293040456, 0), (60656868, 684994, 0), (60474276, 60474284, 0)]
    nodes, edges = graph_funcs.extend_edges(nodes, edges, edges_to_extend)

    # join existing nodes
    new_edge = (60637199, 60637216, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='613'))

    # new edge with osmid
    osmid = 353280664
    new_point = Point(16.370057073009658, 48.22617059933408)
    nodes, edges = graph_funcs.new_edge_osmid_to_point(nodes, edges, osmid, new_point, '613')

    # # join existing nodes
    # new_edge = (199561, 254314477, 0)
    # edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='564'))

    G = ox.graph_from_gdfs(nodes, edges)

    ###### cleanup 3

    # # remove nodes
    del_nodes = [254314477]
    G.remove_nodes_from(del_nodes)

    # get_gdfs
    nodes, edges = ox.graph_to_gdfs(G)

    # join existing nodes
    new_edge = (199561, 199562, 0)
    edges = edges.append(graph_funcs.new_edge_straight(new_edge, nodes, edges, group='564'))

    G = ox.graph_from_gdfs(nodes, edges)

    # saves as graphml
    ox.save_graphml(G, filepath=path.join(gis_data_path, 'streets', 'hand_edit_parallel.graphml'))

    # Save geopackage for import to QGIS
    ox.save_graph_geopackage(G, filepath=path.join(gis_data_path, 'streets', 'hand_edit_parallel.gpkg'), directed=True)

if __name__ == '__main__':
    main()