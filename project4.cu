#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

struct fileinfo {
    int numLines;
    int lineLength;
};

struct matchLocation {
    int x; // row 
    int y; // col
    int pl; // pattern line
};

struct matchCoordinate {
    int x;
    int y;
};


 //Get the fileinfo struct for this file, containing the number of lines and length of a line.

fileinfo getFileInfo(string filename) {
    ifstream lineCounter(filename);
    int numLines = 0;
    string line;
    while (getline(lineCounter, line)) numLines++;
    lineCounter.close();
    struct fileinfo retVal = { numLines, line.length() };
    return retVal;
}


 //Returns the upper leftmost coordinate of a full match, otherwise return null if no full match.

matchCoordinate searchForRealMatches(matchLocation match, matchLocation** allMatches, int* numMatchesArr, int numPatternLines, int world_size) {
    //cout << "-> searchForRealMAtches x:" << match.x << " y: " << match.y << " pl: " << match.pl << endl;
    matchLocation** patternMatchLocations = new matchLocation * [numPatternLines];
    for (int i = 0; i < numPatternLines; i++) patternMatchLocations[i] = nullptr;

    patternMatchLocations[match.pl] = &match;
    for (int i = 0; i < world_size; i++) {
        for (int j = 0; j < numMatchesArr[i]; j++) {
            bool fullMatch = true;
            // for each match...
            if (allMatches[i][j].y == match.y && allMatches[i][j].x != match.x && allMatches[i][j].pl != match.pl) {
                // if it's in the correct column
                for (int k = 0; k < numPatternLines; k++) {
                    if (allMatches[i][j].x == match.x + k && allMatches[i][j].pl == match.pl + k) {
                        // this is a corresponding match!
                        patternMatchLocations[allMatches[i][j].pl] = &allMatches[i][j];
                    }
                    // check if full match 
                    if (patternMatchLocations[k] == nullptr) fullMatch = false;
                }
                if (fullMatch) {
                    struct matchCoordinate retVal = { patternMatchLocations[0]->x, patternMatchLocations[0]->y };
                    delete[] patternMatchLocations;
                    return retVal;
                }
            }
        }
    }
    // return -1, -1 if no match found 
    struct matchCoordinate retVal = { -1, -1 };
    return retVal;
}

__global__ void findPartialMatches(char**inputLines, char**patternLines, int*numInputLines, int*lenInputLines, int*numPatternLines, int*lenPatternLines, int*numMatchesArr, matchLocation**allMatches, int*numThreads) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int totalNumThreads = blockDim.x * (*numThreads);
    numMatchesArr[threadId] = 0; // initialize all numMatches to 0
    if (threadId <= numInputLines) {
        int sizeOfMatchArr = 10; // 10 for now... then dynamically increase if needed
        allMatches[threadId] = new matchLocation[sizeOfMatchArr]; // store the matches at the index for this thread
        for (int j = 0; j < numPatternLines; j++) {
            int pos = 0;
            string jPatternLine(patternLines[j], patternLines[j] + lenPatternLines-1);
            string iInputLine(inputLines[threadId], inputLines[threadId] + lenInputLines-1);
            int found = iInputLine.find(jPatternLine, pos);
            while (found != string::npos) {
                if (numMatchesArr[threadId] >= sizeOfMatchArr) {
                    // increase matchArr size 
                    int biggerSize = sizeOfMatchArr * 2;
                    matchLocation* biggerArr = new matchLocation[biggerSize];
                    memcpy(biggerArr, matchArr, sizeof(matchLocation) * numMatchesArr[threadId]);
                    delete[] matchArr;
                    matchArr = biggerArr;
                    sizeOfMatchArr = biggerSize;
                }
                // store the match
                struct matchLocation m = { threadId, found, j }; // where threadId is the row number, found is the col number, and j is the pattern line number
                matchArr[numMatchesArr[threadId]] = m;
                // update pos, numMatches, and found for the next iteration
                pos = found + 1;
                numMatchesArr[threadId]++; // TODO use atomic cuda increment
                found = iInputLine.find(jPatternLine, pos);
            }
        }
    }
}

int main(int argc, char** argv) {
    // get info about input file
    string inputFile = argv[1];
    fileinfo inputInfo = getFileInfo(inputFile);
    int numInputLines = inputInfo.numLines;
    int lenInputLines = inputInfo.lineLength + 1;

    // read the input file in line by line and store in array
    ifstream file(inputFile);
    char** inputLines = new char* [numInputLines]; // num rows (lines)
    for (int i = 0; i < numInputLines; i++) {
        inputLines[i] = new char[lenInputLines]; // num cols (line length)
    }

    string line;
    int lineNum = 0; // for indexing into the allLines arr
    while (getline(file, line)) {
        strcpy_s(inputLines[lineNum], lenInputLines, line.c_str());
        lineNum++;
    }

    // get info about the pattern file
    string patternFile = argv[2];
    fileinfo patternInfo = getFileInfo(patternFile);
    int numPatternLines = patternInfo.numLines;
    int lenPatternLines = patternInfo.lineLength + 1;

    // read the pattern file in line by line and store in array
    ifstream patternFileStream(patternFile);
    char** patternLines = new char* [numPatternLines]; // num rows (lines)
    for (int i = 0; i < numPatternLines; i++) {
        patternLines[i] = new char[lenPatternLines]; // num cols (line length)
    }
    lineNum = 0; // for indexing into the pattern arr
    while (getline(patternFileStream, line)) {
        strcpy_s(patternLines[lineNum], lenPatternLines, line.c_str());
        lineNum++;
    }

    // allocate memory on device for inputLines and patternLines and copy to the device memory
    char ** inputLinesDevice;
    cudaMalloc(&inputLinesDevice, numInputLines * sizeof(char*)); 
    cudaMemcpy(inputLinesDevice, inputLines, numInputLines * sizeof(char*), cudaMemcpyHostToDevice);
    for (int i = 0; i < numInputLines; i++) {
        cudaMalloc(&inputLinesDevice[i], lenInputLines * sizeof(char));
        cudaMemcpy(inputLinesDevice[i], inputLines[i], lenInputLines * sizeof(char), cudaMemcpyHostToDevice);
    }
    char ** patternLinesDevice;
    cudaMalloc(&patternLinesDevice, numPatternLines * sizeof(char*)); 
    cudaMemcpy(patternLinesDevice, patternLines, numPatternLines * sizeof(char*), cudaMemcpyHostToDevice);
    for (int i = 0; i < numPatternLines; i++) {
        cudaMalloc(&patternLinesDevice[i], lenPatternLines * sizeof(char));
        cudaMemcpy(patternLinesDevice[i], patternLines[i], lenPatternLines * sizeof(char), cudaMemcpyHostToDevice);
    }

    // set the number of blocks and number of threads we want to use
    int numBlocks = 1;
    int numThreads = 32;
    // guessing how to compute this... - maybe for smaller test files we can change this around but for larger it needs to be set
    // if (numInputLines <= 1024) {
    //     numThreads = numInputLines; // according to the internet, 1024 is the max number of threads in a block
    // } else {
    //     numBlocks = ; // ???
    //     numThreads = ; // ???
    // }
    int totalNumThreads = numBlocks * numThreads;

    // setup pointers to get the results from device memory in allMatchLocations and numMatchesArr
    matchLocation** allMatchLocationsDevice;
    cudaMalloc(&allMatchLocationsDevice, numBlocks * sizeof(matchLocation*)); 

    int* numMatchesArrDevice;
    cudaMalloc(&numMatchesArrDevice, numBlocks * sizeof(int)); 
    for (int i = 0; i < numBlocks; i++) {
        cudaMalloc(&numMatchesArrDevice[i], numBlocks * sizeof(int)); 
    }

    // start the kernel to find partial matches
    findPartialMatches<<<numBlocks,numThreads>>>(inputLinesDevice, patternLinesDevice, &numInputLines, &lenInputLines, &numPatternLines, &lenPatternLines, allMatchLocationsDevice, numMatchesArrDevice, &numThreads);

    // copy the results to host memory
    int* numMatchesArr = new int[totalNumThreads];
    matchLocation** allMatchLocations = new matchLocation * [totalNumThreads];

    for (int i = 0; i < totalNumThreads; i++) {
        cudaMemcpy(numMatchesArrDevice[i], numMatchesArr[i], sizeof(int), cudaMemcpyDeviceToHost);
        if (numMatchesArr[i] > 0) {
            cudaMemcpy(allMatchLocationsDevice[i], allMatchLocations[i], numMatchesArr[i] * (matchLocation*), cudaMemcpyDeviceToHost);
        }
    }
    

    // prep the output file 
    ofstream outputFile("output.txt");

    // compare all the matches for lines in a pattern file to find full matches
    int sizeOfCoordArr = 10; // 10 for now... then dynamically increase if needed
    matchCoordinate* coordArr = new matchCoordinate[sizeOfCoordArr];
    int numCoords = 0;
    for (int i = 0; i < numBlocks; i++) {
        for (int j = 0; j < numMatchesArr[i]; j++) {
            matchCoordinate coor = searchForRealMatches(allMatchLocations[i][j], allMatchLocations, numMatchesArr, numPatternLines, world_size);
            if (coor.x == -1 && coor.y == -1) continue; // not a match
            bool alreadyFound = false;
            for (int k = 0; k < numCoords; k++) {
                if (coordArr[k].x == coor.x && coordArr[k].y == coor.y) alreadyFound = true;
            }
            if (alreadyFound) continue; // go to next match
            if (numCoords >= sizeOfCoordArr) {
                // increase coordArr size 
                int biggerSize = sizeOfCoordArr * 2;
                matchCoordinate* biggerArr = new matchCoordinate[biggerSize];
                memcpy(biggerArr, coordArr, sizeof(matchCoordinate) * numCoords);
                delete[] coordArr;
                coordArr = biggerArr;
                sizeOfCoordArr = biggerSize;
            }
            coordArr[numCoords] = coor;
            numCoords++;
            cout << "MATCH AT: " << coor.x << ", " << coor.y << endl;
            // output as column, row and ensure the coordinates are not 0 indexed
            outputFile << coor.y + 1 << ", " << coor.x + 1 << "\n";
        }
    }
    outputFile.close();
    // cleanup memory 
    for (int i = 0; i < numInputLines; i++) {
        cudaFree(inputLinesDevice[i]);
        delete[] inputLines[i];
    }
    delete[] inputLines;
    cudaFree(inputLinesDevice);
    for (int i = 0; i < numPatternLines; i++) {
        cudaFree(patternLinesDevice[i]);
        delete[] patternLines[i];
    }
    delete[] patternLines;
    cudaFree(patternLinesDevice);
    for (int i = 0; i < totalNumThreads; i++) {
        cudaFree(allMatchLocationsDevice[i]);
        delete[] allMatchLocations[i];
    }
    delete[] allMatchLocations;
    cudaFree(allMatchLocationsDevice);
    delete[] numMatchesArr;
    cudaFree(numMatchesArrDevice);
    delete[] coordArr;

    return 0;
}
