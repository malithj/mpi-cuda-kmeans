
typedef struct {
    size_t document_count;
    size_t vocab_count;
    float * document_matrix; 
} DocumentObject;

DocumentObject * get_document_matrix(const float max_df, const int min_df, const char * FILENAME_DOCWORD);
