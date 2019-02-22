#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define METADATA_LEN    25
#define RANK_MASTER     0

/*
    Function type for send/receive used to scatter/gather chunks
*/
typedef void (*Send_Recv_Func)(void **matrix, int start, int end, int width, 
                               int tSize, int src);

/*
    Data type for a colored pixel (with red, green, and blue - RGB)
*/
typedef struct {
    unsigned char r, g, b;
} colorPixel;

/*
    Data type for a grayscale pixel (only with black and white)
*/
typedef struct {
    unsigned char w;
} grayscalePixel;

/*
    The image structure that stores all information needed
*/
typedef struct {
    unsigned char pType[3]; // The type of image as string ( + '\0' at end)
    unsigned char maxVal;   // Max value of a pixel
    unsigned int width;     // The width of image
    unsigned int height;    // The height of image
    void* pixelMatrix;      // Address to the pixel matrix
} image;


/*
    Allocate memory for a matrix of height x width size, having each element
    of sizeT. This is a generic implementation for any type of matrix.
*/
void** createMatrix(unsigned int width, unsigned int height, unsigned int sizeT)
{
    unsigned int i, j;
    void** matrix;

    matrix = (void**)malloc(height * sizeof(void*));
    if (matrix == NULL)
        return NULL;

    for (i = 0; i < height; i++) 
    {
        matrix[i] = malloc(width * sizeT);

        // Error while allocation memory
        if (matrix[i] == NULL)
        {
            for (j = 0; j < i; j++)
                free(matrix[j]);
            return NULL;
        }
    }

    return matrix;
}

/*
    Release the memory of a generic matrix.
*/
void destroyMatrix(void*** matrix, unsigned int height)
{  
    unsigned int i;

    for (i = 0; i < height; i++)
        if ((*matrix)[i] != NULL)
            free((*matrix)[i]);
 
    free(*matrix);
    *matrix = NULL;
}

/*
    Create the pixel matrix, depending on the type of the image (grayscale of 
    colored).
*/
int createPixelMatrix(image *img)
{   
    // For invalid type, do not create a pixel matrix
    if (img == NULL || img->pType[0] != 'P')
        return 0;

    // Identify the type of image
    if (img->pType[1] == '5') 
    {
        // This is a grayscale image
        img->pixelMatrix = createMatrix(img->width, img->height, 
            sizeof(grayscalePixel));
    } 
    else if (img->pType[1] == '6')
    {
        // This is a colored image
        img->pixelMatrix = createMatrix(img->width, img->height, 
            sizeof(colorPixel));
    }

    // Allocation error
    if(!img->pixelMatrix)
        return 0;

    // Success
    return 1;
}

/*
    Read the image form a given file.
*/
void readInput(const char *fileName, image *img) 
{
    unsigned int i, j;
    FILE* inFile;

    // Open the file in binary reading mode
    inFile = fopen(fileName, "rb");
    if (!inFile)
        return;

    // Clear garbage data of image
    memset(img, 0, sizeof(image));
   
    // Read the type from the image file and skip the '\n' character
    fread(img->pType, sizeof(unsigned char), 2, inFile);
    fseek(inFile, 1, SEEK_CUR);

    // Read the width and the height of the image, along with maximul value
    fscanf(inFile, "%d %d\n", &img->width, &img->height);
    fscanf(inFile, "%hhu\n", &img->maxVal);
 
    // Create the matrix of pixels
    if (!createPixelMatrix(img))
        return;

    // Read the matrix of pixels
    if (img->pType[1] == '5') 
    {
        grayscalePixel** matrix = (grayscalePixel**)img->pixelMatrix;
        for (i = 0; i < img->height; i++) 
            for (j = 0; j < img->width; j++)         
                fread(&(matrix[i][j]).w, sizeof(unsigned char), 1, inFile);
    } 
    else if (img->pType[1] == '6') 
    {
        colorPixel** matrix = (colorPixel**)img->pixelMatrix;
        for (i = 0; i < img->height; i++) 
            for (j = 0; j < img->width; j++) 
                fread(&(matrix[i][j]).r, sizeof(unsigned char), 3, inFile);
    }

    // Close the file
    fclose(inFile);
}

/*
    Write the image to file.
*/
void writeData(const char *fileName, image *img) 
{
    unsigned int i, j;
    FILE* outFile;

    // Open the file in binary writing mode
    outFile = fopen(fileName, "wb");
    if (!outFile)
        return;
    
    // Write image metadata to file
    fprintf(outFile, "%c%c\n", (img->pType)[0], (img->pType)[1]);
    fprintf(outFile, "%d %d\n", img->width, img->height);
    fprintf(outFile, "%d\n", img->maxVal);

    // Read the matrix of pixels
    if (img->pType[1] == '5')
    {
        grayscalePixel** matrix = (grayscalePixel**)img->pixelMatrix;
        for (i = 0; i < img->height; i++) 
            for (j = 0; j < img->width; j++) 
                fwrite(&matrix[i][j].w, sizeof(unsigned char), 1, outFile);
    } 
    else if (img->pType[1] == '6')
    {
        colorPixel** matrix = (colorPixel**)img->pixelMatrix;
        for (i = 0; i < img->height; i++) 
            for (j = 0; j < img->width; j++) 
                fwrite(&matrix[i][j].r, sizeof(unsigned char), 3, outFile);
    }

    // Close the file and release memory for output image
    fclose(outFile);
}

/*
    This function computes the size of the interval [0, N - 1] divided to each
    process of P processes. The rank identifies a process. This function divide
    the interval in a fairly manner; each interval has a difference in size of 
    maximum 1 element.
*/
int intervalSize(int N, int P, int rank)
{
    int size = N / P;
    int remainder = N % P;
    int nextRank = rank + 1;

    int start = rank > remainder ? remainder + size * rank : (size + 1) * rank;
    int end = nextRank > remainder ? remainder + size * nextRank : 
              (size + 1) * nextRank;

    return end - start;
}

/*
    This is a "primitive" used to send a chunk of matrix from 'start' 
    (inclusive) to 'end' (exclusive). Each line has 'width' elements of size 
    'tSize'. The destination is 'dst'.
*/
void sendChunk(void **matrix, int start, int end, int width, int tSize, int dst)
{
    int l;
    int chunkHeight = end - start;
    int chunkWidth = width * tSize;
    char *buf = (char*)malloc(chunkHeight * chunkWidth * sizeof(char));
    if (!buf)
        return;
    
    for (l = 0; l < chunkHeight; l++)
        memcpy(buf + l * chunkWidth, ((void**)matrix)[start + l], chunkWidth);

    MPI_Send(buf, chunkHeight * chunkWidth, MPI_CHAR, dst, 0, MPI_COMM_WORLD);
    
    free(buf);
}

/*
    This is a "primitive" used to receive a chunk of matrix and put it from 
    'start' (inclusive) to 'end' (exclusive). Each line has 'width' elements of 
    size 'tSize'. The destination is 'src'.
*/
void recvChunk(void **matrix, int start, int end, int width, int tSize, int src)
{
    int l;
    int chunkHeight = end - start;
    int chunkWidth = width * tSize;
    char *buf = (char*)malloc(chunkHeight * chunkWidth * sizeof(char));
    if (!buf)
        return;

    MPI_Recv(buf, chunkHeight * chunkWidth, MPI_CHAR, src, 0, MPI_COMM_WORLD, 
             MPI_STATUS_IGNORE);

    for (l = 0; l < chunkHeight; l++) 
        memcpy(((void**)matrix)[start + l], buf + l * chunkWidth, chunkWidth);

    free(buf);
}

/*
    This function is a generic implementation of a Gather/Scatter, but for 
    matrices. From each process is sent/received a chunk of memory. The 
    action of this function is defined by the 'func' parameter which may be
    sendChunk (Scatter) or recvChunk (Gather).
*/
void sendRecvChunk(void** matrix, int height, int width, int tSize, 
                   int nProcesses, int crt_rank, Send_Recv_Func func) 
{
    int i;
    int crtHeight = 0;

    for (i = 0; i < nProcesses; i++) 
    {
        int chunkSize = intervalSize(height, nProcesses, i);
        int start = crtHeight;
        crtHeight += chunkSize;

        if (i == crt_rank)
            continue;

        func(matrix, start, start + chunkSize, width, tSize, i);
    }
}

/*
   This function applies a filter to the inMatrix and puts the result in matrix.
   The filter is applied on each channel (determined by tSize). For the first 
   and the last rows from the original image, the filter is not applied. 
*/
void applyOneFilter(void** matrix, int height, int width, int tSize, int rank, 
                    int nProcesses, float K_matrix[3][3], float K_const, 
                    int iteration, void** inMatrix)
{
    int i, j, s, t, color;
    float sum;
     
    // Apply the filter to each pixel of the image    
    for (i = 1; i < height + 1; i++)
        for (j = 1; j < width - 1; j++)
        {
            if ((i == 1 && rank == 0) 
                || (i == height && rank == nProcesses - 1))
            {
                continue;
            }
            else 
            {
                // Apply the filter for each component
                for (color = 0; color < tSize; color++)
                { 
                    // Init partial sum to be 0
                    sum = 0;

                    // Compute the sum
                    for (s = 0; s < 3; s++) 
                        for (t = 0; t < 3; t++) 
                        {
                            unsigned char *line = inMatrix[i + s - 1];
                            int index = (j + t - 1) * tSize + color;
                            sum += (float)line[index] * K_matrix[s][t];
                        }

                    // Put the color in the matrix
                    int jIdx = j * tSize + color;
                    ((unsigned char*)matrix[i - 1])[jIdx] = (unsigned char)sum;
                }
            }
        }
}

/*
    This function solves the boundaries problems - the lines from the original
    image that are between processes. These line needed by 2 processes are sent
    and received. Then, the matrix is copied and ready to be used.
*/
void solveBoundaries(void** matrix, int height, int width, int tSize, 
                     int rank, int nProcesses, void** inMatrix) 
{
    int l;

    // Send and receive data to other processes
    if (rank > 0) 
        sendChunk(matrix, 0, 1, width, tSize, rank - 1);

    if (rank < nProcesses - 1) 
        recvChunk(inMatrix, height + 1, height + 2, width, tSize, rank + 1);

    if (rank < nProcesses - 1) 
        sendChunk(matrix, height - 1, height, width, tSize, rank + 1);

    if (rank > 0) 
        recvChunk(inMatrix, 0, 1, width, tSize, rank - 1);

    // Copy the image in inMatrix
    for (l = 0; l < height; l++) 
        memcpy(inMatrix[l + 1], matrix[l], width * tSize);
}

/*
   This function applied all filters to a matrix of pixels. The filters are 
   predefined (smooth, blur, sharpe, mean and emboss).
*/
void applyFilters(void** matrix, int height, int width, int tSize, int rank, 
                  int nProcesses, int numFilters, char **filters)
{
    int i, j, k;
    
    // Initialize filters matrices and constants
    float K_matrix[5][3][3] = {
        {{ 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}},
        {{ 1,  2,  1}, { 2,  4,  2}, { 1,  2,  1}},
        {{ 0, -2,  0}, {-2, 11, -2}, { 0, -2,  0}},
        {{-1, -1, -1}, {-1,  9, -1}, {-1, -1, -1}},
        {{ 0,  1,  0}, { 0,  0,  0}, { 0, -1,  0}}
    };
    float K_const[5] = { 9, 16, 3, 1, 1 };
    
    // Prepare filter to be used
    for (k = 0; k < 5; k++)
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                K_matrix[k][i][j] /= K_const[k];

    // Create a copy of the image
    void** inMatrix = createMatrix(width, height + 2, tSize);

    // Apply each filter
    for (i = 0; i < numFilters; i++) 
    {
        solveBoundaries(matrix, height, width, tSize, rank, nProcesses, 
                        inMatrix);
        
        if (!strcmp(filters[i], "smooth")) 
            applyOneFilter(matrix, height, width, tSize, rank, nProcesses, 
                           K_matrix[0], K_const[0], i, inMatrix);
        else if (!strcmp(filters[i], "blur")) 
            applyOneFilter(matrix, height, width, tSize, rank, nProcesses, 
                           K_matrix[1], K_const[1], i, inMatrix);
        else if (!strcmp(filters[i], "sharpen")) 
            applyOneFilter(matrix, height, width, tSize, rank, nProcesses, 
                           K_matrix[2], K_const[2], i, inMatrix);
        else if (!strcmp(filters[i], "mean")) 
            applyOneFilter(matrix, height, width, tSize, rank, nProcesses, 
                           K_matrix[3], K_const[3], i, inMatrix);
        else if (!strcmp(filters[i], "emboss")) 
            applyOneFilter(matrix, height, width, tSize, rank, nProcesses, 
                           K_matrix[4], K_const[4], i, inMatrix);
    }

    // Destroy the copy
    destroyMatrix(&inMatrix, height);
}

/*
    This is the main function executed by the master process. It reads the 
    image, broadcasts the infos about the image, sends chunks, applies the 
    the filters, receives the chunks processed, reassembles the image and 
    writes it to the file system.
*/
void masterProcess(int argc, char **argv, int rank, int nProcesses)
{
    int l;

    // Process program arguments
    char *inFile = argv[1];
    char *outFile = argv[2];
    char **filters = &(argv[3]);
    int numFilters = argc - 3;

    // Read the image from the file
    image img;
    readInput(inFile, &img);

    // Broadcast info about image to all processes
    char metadata[METADATA_LEN];
    sprintf(metadata, "%s %c %d %d", img.pType, img.maxVal, img.width, 
            img.height);

    MPI_Bcast(&metadata, METADATA_LEN, MPI_CHAR, RANK_MASTER, MPI_COMM_WORLD);

    // Compute the size of chunk and allocate memory for the chunk of master
    void** chunk;
    int tSize = (img.pType[1] == '5') ? sizeof(grayscalePixel) : 
                (img.pType[1] == '6') ? sizeof(colorPixel) : 0;
    int chunkHeight = intervalSize(img.height, nProcesses, rank);
    int chunkWidth = img.width;

    chunk = createMatrix(chunkWidth, chunkHeight, tSize);

    // Copy top part of the image in chunk
    for (l = 0; l < chunkHeight; l++) 
        memcpy(chunk[l], ((void**)img.pixelMatrix)[l], chunkWidth * tSize);

    // Scatter chunks of images to each process
    sendRecvChunk(img.pixelMatrix, img.height, chunkWidth, tSize, nProcesses, 
                  rank, sendChunk);

    // Apply filters
    applyFilters(chunk, chunkHeight, chunkWidth, tSize, rank, nProcesses, 
                 numFilters, filters);

    // Getter the chunks from slave processes
    sendRecvChunk(img.pixelMatrix, img.height, chunkWidth, tSize, nProcesses, 
                  rank, recvChunk);
    
    // Copy top part image modified to the img
    for (l = 0; l < chunkHeight; l++) 
        memcpy(((void**)img.pixelMatrix)[l], chunk[l], chunkWidth * tSize);

    // Write the image to the output file
    writeData(outFile, &img);

    // Destroy the image
    destroyMatrix((void***)&(img.pixelMatrix), img.height);
}

/*
    This is the main function run by a slave process. It receives the metadata
    and the chunk to be processed, processes that chunk by appling the filters
    and then sends that processed chunk to the master process.
*/
void slaveProcess(int argc, char **argv, int rank, int nProcesses)
{
    char **filters = &(argv[3]);
    int numFilters = argc - 3;

    char metadata[METADATA_LEN], pType[3], maxVal;
    int width, height;

    // Broadcast info about image to all processes
    MPI_Bcast(&metadata, METADATA_LEN, MPI_CHAR, RANK_MASTER, MPI_COMM_WORLD);
    sscanf(metadata, "%s %c %d %d", pType, &maxVal, &width, &height);

    // Compute the size of chunk and allocate memory for the chunk of master
    void** chunk;
    int tSize = (pType[1] == '5') ? sizeof(grayscalePixel) : 
                (pType[1] == '6') ? sizeof(colorPixel) : 0;
    int chunkHeight = intervalSize(height, nProcesses, rank);
    int chunkWidth = width;

    chunk = createMatrix(chunkWidth, chunkHeight, tSize);

    // Receive chunks of images from master process
    recvChunk(chunk, 0, chunkHeight, chunkWidth, tSize, RANK_MASTER);

    // Apply filters
    applyFilters(chunk, chunkHeight, chunkWidth, tSize, rank, nProcesses, 
                 numFilters, filters);

    // Send the chunks of image to master
    sendChunk(chunk, 0, chunkHeight, chunkWidth, tSize, RANK_MASTER);
}

/*
    The main function that initializes the MPI and calls the right "main" 
    function for each process type (master of slave).
*/
int main(int argc, char **argv) 
{
    int rank, nProcesses;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

    if (rank == RANK_MASTER) 
        masterProcess(argc, argv, rank, nProcesses);
    else 
        slaveProcess(argc, argv, rank, nProcesses);

    MPI_Finalize();
    return 0;
}