#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../preprocess/include/txtproc.h"
#include "../preprocess/include/cnfparser.h"

#define THRESHOLD 0.0001

extern void do_cluster_on_gpu(const float * global_vectors, const int total_number,
                              const int num_of_clusters, const int dimension); 

float * global_vectors;
          
const char * CONFIG_FILE = "resources/config.CONFIG";

int main() {
    int N = 0;   /* Total number of samples */
    int k = 8;   /* Number of clusters */
    int d = 0;   /* Dimensions of data */
    
    /* read the configuration file */
    char ** config = get_config(CONFIG_FILE);
    
    /* init global vector */
    DocumentObject * dom = get_document_matrix(0.5, 2, config);
    N = dom->document_count;
    k = dom->cluster_count;
    d = dom->vocab_count;
    global_vectors = dom->document_matrix; 
    
    /* run gpu code */
    do_cluster_on_gpu(global_vectors, N, k, d);
    free(global_vectors);
}
