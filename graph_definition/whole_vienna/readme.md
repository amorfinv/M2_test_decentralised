First step is to run "create_graph.py"
Second step is to run "graph_edit.py"
Third step is to perform manual edits (hand_edits.gpkg)
Fourth step is to run "cleanup_graph.py"
Fifth step is to perform second round of manual edits (hand_edits_2.gpkg)
Sixth step is to run "cleanup_graph_2.py"


TODO: fix edge (199626, 33301552)!!!

Now do a first separating group where you merge/split groups
group_split1.py separator groups:
136,139,85,117,1561,1562,1,32,19,20,1003,1017,1571,127,8,1573,0

using group_split1 regrouping_smart.gpkg you can divide the airspace into zones.

Now do a refined group_split that consideres group_split1.py and the new airspace zones 
-group_split2.py separator groups:
136,139,85,1,18,0

