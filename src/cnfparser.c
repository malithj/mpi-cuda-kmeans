#include "include/cnfparser.h"

const char ** get_config(const char * filename) {
    char * line;
    FILE * file;
    const char ** results = (const char **)calloc(2, sizeof(const char *));

    int read       = 0;
    size_t len     = 0;
        
    char * prop  = (char *)calloc(200, sizeof(char));
    char * value = (char *)calloc(200, sizeof(char)); 
    
    file = fopen(filename, "r");

    if (file == NULL) {
        perror("Error opening the config file \n");
        exit(EXIT_FAILURE);
    }
    
    printf("Reading properties from file: %s\n", filename);
    printf("-----------------------------------------------------\n");
    while ((read = getline(&line, &len, file)) != -1) {

        if(line[0] == '#') continue;

        if(sscanf(line, "%s = %s", prop, value) != 2) {
            fprintf(stderr, "Syntax error, line \n");
            continue;
        }
        if (strcmp(prop, "FILENAME_DOCWORD") == 0) {
            results[0] = value;
        }
        if (strcmp(prop, "FILENAME_VOCAB") == 0) {
            results[1] = value;
        }
        printf("%s = %s \n", prop, value);
        prop  = (char *)calloc(200, sizeof(char));
        value = (char *)calloc(200, sizeof(char)); 
    }
    return results;
}
