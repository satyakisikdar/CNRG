import csv
import sys

def get_names(cluster_filename, metadata_filename):
    ''' Given a cluster output from motifcluster, print out the names of
    the nodes in each cluster. '''

    index_nodeid_map = {}
    with open(metadata_filename, 'rb') as metadata:
        reader = csv.DictReader(metadata)
        for row in reader:
            index_nodeid_map[int(row["node_id"])] = row["name"]

    with open(cluster_filename) as clusters:
        for cluster in clusters:
            node_ids = [int(index) for index in cluster.strip().split('\t')]
            names = [index_nodeid_map[nid] for nid in node_ids]
            print '\t'.join(names)
            
if __name__ == '__main__':
    try:
        cluster_filename = sys.argv[1]
        metadata_filename = sys.argv[2]
    except:
        sys.stderr.write('USAGE: python get_names.py cluster_filename metadata_filename\n')        
    get_names(cluster_filename, metadata_filename)
