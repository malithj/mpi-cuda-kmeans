#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/txtproc.h"

//const char * FILENAME_DOCWORD = "../bag_of_words/enron/docword.enron.txt";
//const char * FILENAME_VOCAB   = "../bag_of_words/enron/vocab.enron.txt";

float * document_matrix;
int   * df_array;
float * idf_array;
float * sorted_idf_array;
int   * index_map;

int compare(const void *a,const void *b) {
    float *x = (float *) a;
    float *y = (float *) b;
    if (*x < *y) return 1;
    if (*x > *y) return -1;
     return 0;
}

DocumentObject * get_document_matrix(const float max_df, const int min_df, char ** config) {
    char * FILENAME_DOCWORD = config[0];

    FILE * docword;   /* define document word count file */
    FILE * vocab;     /* define vocabulary */
  
    char * delim          = (char *)malloc(sizeof(char));
    * delim               = ' ';
    char * line           = NULL;
    int    read           = 0;
    size_t len            = 0;
    size_t count          = 0;
    size_t document_count = 0;
    size_t vocab_count    = 0;
    size_t doc_lines      = 0;
    size_t cluster_count  = atoi(config[3]); 
    int max_vocab         = atoi(config[2]);

    docword = fopen(FILENAME_DOCWORD, "r");

    if (docword == NULL) {
        perror("Error while opening the document word count file ");
        exit(EXIT_FAILURE);
    } 
    
    while ((read = getline(&line, &len, docword)) != -1) {
        line[strlen(line) - 1] = '\0';
        switch(count) {
           case 0:
               document_count = atoi(line);  
               break; 
 
           case 1:
               vocab_count = atoi(line);  
               break;          

           case 2:
               doc_lines = atoi(line);  
               break;             
        }

        if (count == 2 && document_count > 0 && vocab_count > 0) {
            df_array = (int *)malloc(vocab_count * sizeof(int));
            idf_array = (float *)malloc(vocab_count * sizeof(float));
            sorted_idf_array = (float *)malloc(vocab_count * sizeof(float));
            index_map = (int *)malloc(vocab_count * sizeof(int));
            for (int i = 0; i < vocab_count; i ++) {
                df_array[i] = 0;
                index_map[i] = 0;
            }
        } 
        count += 1;
        if (count > 3) {
            size_t * value      =(size_t *) malloc(3 * sizeof(size_t));
            size_t token_number = 0;
            char * ptr = strtok(line, delim);
            while(ptr != NULL) {
                value[token_number ++]  = atoi(ptr);
		ptr = strtok(NULL, delim);
            }

            size_t doc_id     = value[0] - 1;
            size_t word_id    = value[1] - 1;
            size_t word_count = value[2];
            df_array[word_id] += 1;
        }
    }
      
    rewind(docword);

    for (int i = 0; i < vocab_count; i ++) {
        if (df_array[i] != 0) {
            idf_array[i] = (1 / (float)(df_array[i]));
        }
    }

    for (size_t i = 0; i < vocab_count; i ++) {
        sorted_idf_array[i] = idf_array[i];
    }
    
    qsort(sorted_idf_array, vocab_count, sizeof(sorted_idf_array[0]), compare);
    
    float min_idf = max_vocab == -1 ? 0 : sorted_idf_array[max_vocab];
    
    size_t max_count = (size_t)(max_df * document_count);
    size_t index     = 0;
    for (int i = 0; i < vocab_count; i ++) {
        if (df_array[i] > max_count || df_array[i] < min_df || idf_array[i] <= min_idf) {
            index_map[i] = -1;
        } else {
            index_map[i] = index;
            index ++;
        }
    }

    count = 0;
    while ((read = getline(&line, &len, docword)) != -1) {
        line[strlen(line) - 1] = '\0';

        if (count == 2 && document_count > 0 && vocab_count > 0) {
            document_matrix = (float *)malloc(document_count * index * sizeof(float));
        } 
        count += 1;
        if (count > 3) {
            size_t * value      = (size_t *)malloc(3 * sizeof(size_t));
            size_t token_number = 0;
            char * ptr = strtok(line, delim);
            while(ptr != NULL) {
                value[token_number ++]  = atoi(ptr);
		ptr = strtok(NULL, delim);
            }

            size_t doc_id     = value[0] - 1;
            size_t word_id    = index_map[value[1] - 1];
            size_t word_count = value[2];
            if (word_id != -1) {
                document_matrix[doc_id * index + word_id] = (float)word_count; 
            }
        }
    } 
    printf("Completed reading file: %s\n", FILENAME_DOCWORD);
    printf("Preprocessed vocabulary count: %d\n", index); 
    DocumentObject * dom;
    dom = (DocumentObject *)malloc(sizeof(DocumentObject));
    dom -> document_count = document_count;
    dom -> cluster_count = cluster_count;
    dom -> vocab_count    = index;
    dom -> document_matrix = document_matrix;
    free(line);
    free(index_map);
    free(df_array);
    free(delim);
    fclose(docword);
    return dom;
}

