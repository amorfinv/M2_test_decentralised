-------------------------initial prep-------------------------------------
First step is to run "create_graph.py"
Second step is to run "graph_edit.py"
Third step is to perform manual edits (hand_edits.gpkg)
Fourth step is to run "cleanup_graph.py"
Fifth step is to perform second round of manual edits (hand_edits_2.gpkg)
Sixth step is to run "cleanup_graph_2.py"

Even after cleanup_graph2.py, there are still dead-ends to remove so we remove
the dead-ends prior to continuiing anything else.

    # delete some dead ends
    nodes_to_remove = [2107289289, 111166066, 297732617, 319898633, 117385463, 293431182, 2273750271,
                      280697142, 309367640, 7867573020, 274610745, 1676206186, 3573081082, 2034593346,
                      33301993, 8307089679, 283734344, 272446352, 5120275681, 2200458285, 199625, 252281626, 295431612,
                      299084684, 3534479782, 1829890434, 130232679, 43511494, 130232678, 378477, 4032457020, 4032457019,
                      4032457015]
    G.remove_nodes_from(nodes_to_remove)


---------------------graph for genetic algotihm--------------------------
prep_graph_gen.py takes a graph and creates two different graphs to direct
graph_dir_0 and graph_dir_1. Currently this was with cleanup_graph_2. plus dead ends

graph_builder.py takes graph_dir_0 and graph_dir_1 so that direction can
be set easily in the gen_algorithm.

Also after running the gen algorithm we can use graph_builder.py and the results of gen algo
to redirect the graph based on the results of the genetic algorithm.

Parse genetic algorithm to get best results with parse_gen_results.py.

--------------------------edits after genetic algorithm---------------
After running the genetic algorithm we performed with cleanup_graph_2.py
plus some dead_end removals we try and start removing some parallel streets.

This starts with gen_edits,py which creates regrouping_3

-------------------------------regrouping------------------------------
group_split1.py (old)
group_split2.py (prior to genetic algorithm). Runs with cleanup_graph_2 plus dead end removals

create_separators.py groups separator after regrouping_3

-----------------------------get_center_points-----------------------
This takes a graph and gets the center points of all the edges so that
they can be used as origin/desitnations.

----------------------------Airspace zones----------------------------
airspace_zones.py This is used to create the airspace zones.

---------------------------parallel streets------------------------
identify_parallel_streets.py can take any graph adn return how close things are

---------------------------Some more edits--------------------------
Take regrouping_3 and start removing all parallel streets
