#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/txtproc.h"

const char * FILENAME_DOCWORD = "../bag_of_words/enron/docword.enron.txt";
const char * FILENAME_VOCAB   = "../bag_of_words/enron/vocab.enron.txt";

float * document_matrix;
int   * df_array;

DocumentObject * get_document_matrix(const float max_df, const int min_df) {
    FILE * docword;   /* define document word count file */
    FILE * vocab;     /* define vocabulary */
  
    char * delim          = " ";
    char * line           = NULL;
    int    read           = 0;
    size_t len            = 0;
    size_t count          = 0;
    size_t document_count = 0;
    size_t vocab_count    = 0;
    size_t doc_lines      = 0;

    docword = fopen(FILENAME_DOCWORD, "r");
    vocab   = fopen(FILENAME_VOCAB, "r");

    if (docword == NULL || vocab == NULL) {
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

            for (int i = 0; i < vocab_count; i ++) {
                df_array[i] = 0;
            }
        } 
        count += 1;
        if (count > 3) {
            size_t * value      = malloc(3 * sizeof(size_t));
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

    size_t max_count = (size_t)(max_df * document_count);
    size_t index     = 0;
    for (int i = 0; i < vocab_count; i ++) {
        if (df_array[i] > max_count || df_array[i] < min_df) {
            df_array[i] = -1;
        } else {
            df_array[i] = index;
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
            size_t * value      = malloc(3 * sizeof(size_t));
            size_t token_number = 0;
            char * ptr = strtok(line, delim);
            while(ptr != NULL) {
                value[token_number ++]  = atoi(ptr);
		ptr = strtok(NULL, delim);
            }

            size_t doc_id     = value[0] - 1;
            size_t word_id    = df_array[value[1] - 1];
            size_t word_count = value[2];
            document_matrix[doc_id * index + word_id] = (float)word_count; 
        }
    }
    printf("Completed reading file: %s\n", FILENAME_DOCWORD);
    fclose(docword);
    fclose(vocab);
    DocumentObject * dom;
    dom = (DocumentObject *)malloc(sizeof(DocumentObject));
    dom -> document_count = document_count;
    dom -> vocab_count    = index;
    dom -> document_matrix = document_matrix;
    return dom;
}

