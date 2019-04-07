#include "include/cnfparser.h"

char ** get_config(const char * filename) {
    char * line;
    FILE * file;
    char ** results = (char **)calloc(4, sizeof(char *));
    const char * delim = " = ";

    int read       = 0;
    size_t len     = 0;

    char * prop    = (char *)calloc(100, sizeof(char));
    char * value   = (char *)calloc(100, sizeof(char)); 
    char * token;

    file = fopen(filename, "r");

    if (file == NULL) {
        perror("Error opening the config file \n");
        exit(EXIT_FAILURE);
    }
    
    printf("Reading properties from file: %s\n", filename);
    printf("-----------------------------------------------------\n");
    while ((read = getline(&line, &len, file)) != -1) {

        if(line[0] == '#') continue;

        token = strtok(line, delim);
        size_t count = 0;
        while (token != NULL) {
            if (count == 0) {
                strcpy(prop, token);
            } else {
                strcpy(value, token);
                value[strlen(value) - 1] = '\0';
            }
            token = strtok(NULL, delim);
            count += 1;
        }
        if (strcmp(prop, "FILENAME_DOCWORD") == 0) {
            results[0] = value;;
        }
        if (strcmp(prop, "FILENAME_VOCAB") == 0) {
            results[1] = value;
        }
        if (strcmp(prop, "MAX_VOCAB") == 0) {
            results[2] = value;
        }
        if (strcmp(prop, "NUM_CLUSTERS") == 0) {
            results[3] = value;
        }
        printf("%s = %s \n", prop, value);
        prop  = (char *)calloc(100, sizeof(char));
        value = (char *)calloc(100, sizeof(char)); 
    }
    fclose(file);
    free(line);
    free(prop);
    free(value);
    return results;
}
