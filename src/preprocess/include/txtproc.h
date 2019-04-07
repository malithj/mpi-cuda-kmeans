#define _GNU_SOURCE

typedef struct {
    size_t document_count;
    size_t vocab_count;
    size_t cluster_count;
    float * document_matrix; 
} DocumentObject;

DocumentObject * get_document_matrix(const float max_df, const int min_df, char ** config);
