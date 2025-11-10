java -cp elki-bundle-0.8.0.jar elki.application.KDDCLIApplication \
-dbc.in /home/shukurillo/lab/OD/autoglosh-revisited/datasets/toy/toy.csv \
-dbc.parser NumberVectorLabelParser \
-algorithm outlier.clustering.GLOSH \
-hdbscan.minPts 3 \
-hdbscan.minclsize 3 \
-resulthandler ResultWriter \
-out out/

