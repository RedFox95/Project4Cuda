#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

struct fileinfo {
    int numLines;
    size_t lineLength;
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

matchCoordinate searchForRealMatches(matchLocation match, matchLocation* allMatches, int numMatches, int numPatternLines, int totalNumThreads) {
    //cout << "-> searchForRealMAtches x:" << match.x << " y: " << match.y << " pl: " << match.pl << endl;
    matchLocation** patternMatchLocations = new matchLocation * [numPatternLines];
    for (int i = 0; i < numPatternLines; i++) patternMatchLocations[i] = nullptr;

    patternMatchLocations[match.pl] = &match;
    for (int i = 0; i < numMatches; i++) {
        bool fullMatch = true;
        // for each match...
        if (allMatches[i].y == match.y && allMatches[i].x != match.x && allMatches[i].pl != match.pl) {
            // if it's in the correct column
            for (int k = 0; k < numPatternLines; k++) {
                if (allMatches[i].x == match.x + k && allMatches[i].pl == match.pl + k) {
                    // this is a corresponding match!
                    patternMatchLocations[allMatches[i].pl] = &allMatches[i];
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
    // return -1, -1 if no match found 
    struct matchCoordinate retVal = { -1, -1 };
    return retVal;
}

__device__ void findSubStr(char*str, int row, int strLen, char* subStr, int patternLineNum, int subStrLen, int pos, int*foundPos) {
    printf("-> findSubStr row %d, strLen %d, patternLineNum %d, subStrLen %d, pos %d\n", row, strLen, patternLineNum, subStrLen, pos);
    *foundPos = -1; // assume no match found
    for (int i = pos; i <= strLen - subStrLen; i++) {
        //printf("r %d: i is %d\n", row, i);
        bool found = true;
        for (int j = 0; j < subStrLen; j++) {
            //printf("r %d: j is %d and i+j is %d comparing %c vs %c\n", row, j, i + j, str[(row * strLen) + i + j], subStr[(patternLineNum * subStrLen)  + j]);
            if (str[(row*strLen) + i + j] != subStr[(patternLineNum * subStrLen)+j]) {
                found = false;
                break;
            }
        }
        if (found) {
            *foundPos = i;
            //printf("r %d: FOUND match in device foundPos is %d\n", row, *foundPos);
            break;
        }
    }
}

__global__ void findPartialMatches(char* inputLines, char* patternLines, int* numInputLines, int* lenInputLines, int* numPatternLines, int* lenPatternLines, int* numMatches, matchLocation* allMatches) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int threadIndex = threadId * *lenInputLines;
    //printf("Tmg thread id is %d and index is %d\n", threadId, threadIndex);
    if (threadId < *numInputLines) {
        for (int j = 0; j < *numPatternLines; j++) {
            int pos = 0;
            int found = -1;
            findSubStr(inputLines, threadId, *lenInputLines, patternLines, j, *lenPatternLines, 0, &found);
            while (found != -1) {
                // store the match
                struct matchLocation m = { threadId, found, j }; // where threadId is the row number, found is the col number, and j is the pattern line number
                int storingIndex = atomicAdd(numMatches, 1);
                allMatches[storingIndex] = m;
                // update pos, numMatches, and found for the next iteration
                pos = found + 1;
                findSubStr(inputLines, threadId,  *lenInputLines, patternLines, j, *lenPatternLines, pos, &found);
            }
        }
    }
}

int main(int argc, char** argv) {
    // get info about input file
    string inputFile = argv[1];
    fileinfo inputInfo = getFileInfo(inputFile);
    int numInputLines = inputInfo.numLines;
    int lenInputLines = inputInfo.lineLength;

    // read the input file in line by line and store in array
    ifstream file(inputFile);
    char** inputLines = new char* [numInputLines]; // num rows (lines)
    for (int i = 0; i < numInputLines; i++) {
        inputLines[i] = new char[lenInputLines+1]; // num cols (line length)
    }

    // temp test..
    char* inputLine1d = new char[numInputLines * lenInputLines];

    string line;
    int lineNum = 0; // for indexing into the allLines arr
    while (getline(file, line)) {
        strcpy_s(inputLines[lineNum], lenInputLines+1, line.c_str());
        for (int i = 0; i < lenInputLines; i++) {
            inputLine1d[(lineNum * lenInputLines) + i] = line.c_str()[i];
        }  
        lineNum++;
    }
    // get info about the pattern file
    string patternFile = argv[2];
    fileinfo patternInfo = getFileInfo(patternFile);
    int numPatternLines = patternInfo.numLines;
    int lenPatternLines = patternInfo.lineLength;

    // read the pattern file in line by line and store in array
    ifstream patternFileStream(patternFile);
    char** patternLines = new char* [numPatternLines]; // num rows (lines)
    for (int i = 0; i < numPatternLines; i++) {
        patternLines[i] = new char[lenPatternLines+1]; // num cols (line length)
    }
    char* patternLine1d = new char[numPatternLines*lenPatternLines];

    lineNum = 0; // for indexing into the pattern arr
    while (getline(patternFileStream, line)) {
        strcpy_s(patternLines[lineNum], lenPatternLines+1, line.c_str());
        for (int i = 0; i < lenPatternLines; i++) {
            patternLine1d[(lineNum * lenPatternLines) + i] = line.c_str()[i];
        }
        lineNum++;
    }
    // allocate memory on device for inputLines and patternLines and copy to the device memory
    char* inputLinesDevice;
    cudaMalloc((void**)&inputLinesDevice, lenInputLines * numInputLines * sizeof(char)); // inputLinesDevice will be the 2d array flattened 
    cudaMemcpy((void*)inputLinesDevice, (void*)inputLine1d, lenInputLines * numInputLines * sizeof(char), cudaMemcpyHostToDevice);
    char* patternLinesDevice;
    cudaMalloc((void**)&patternLinesDevice, lenPatternLines * numPatternLines * sizeof(char));
    cudaMemcpy((void*)patternLinesDevice, (void*)patternLine1d, lenPatternLines * numPatternLines * sizeof(char), cudaMemcpyHostToDevice);

    //allocate memory for the length and num of lines for both files 
    int* numInputLinesDevice;
    cudaMalloc((void**) & numInputLinesDevice, sizeof(int));
    cudaMemcpy(numInputLinesDevice, & numInputLines, sizeof(int), cudaMemcpyHostToDevice);
    int* lenInputLinesDevice;
    cudaMalloc((void**) & lenInputLinesDevice, sizeof(int));
    cudaMemcpy((void*)lenInputLinesDevice, &lenInputLines, sizeof(int), cudaMemcpyHostToDevice);
    int* numPatternLinesDevice;
    cudaMalloc((void**) & numPatternLinesDevice, sizeof(int));
    cudaMemcpy((void*)numPatternLinesDevice, &numPatternLines, sizeof(int), cudaMemcpyHostToDevice);
    int* lenPatternLinesDevice;
    cudaMalloc((void**) & lenPatternLinesDevice, sizeof(int));
    cudaMemcpy((void*)lenPatternLinesDevice, &lenPatternLines, sizeof(int), cudaMemcpyHostToDevice);

    // set the number of blocks and number of threads we want to use
    int numBlocks = 1;
    int numThreads = 10;
    // guessing how to compute this... - maybe for smaller test files we can change this around but for larger it needs to be set
    // if (numInputLines <= 1024) {
    //     numThreads = numInputLines; // according to the internet, 1024 is the max number of threads in a block
    // } else {
    //     numBlocks = ; // ???
    //     numThreads = ; // ???
    // }
    int totalNumThreads = numBlocks * numThreads;

    // setup pointers to get the results from device memory in allMatchLocations and numMatchesArr
    matchLocation* allMatchLocationsDevice;
    cudaMalloc(&allMatchLocationsDevice, 9999 * sizeof(matchLocation)); // temp.. limiting to 9999 partial matches
    int* numMatchesArrDevice;
    cudaMalloc(&numMatchesArrDevice, sizeof(int));
    cudaMemset(numMatchesArrDevice, 0, 1); // initialize numMatches to 0

    // start the kernel to find partial matches
    findPartialMatches<<<numBlocks,numThreads>>>(inputLinesDevice, patternLinesDevice, numInputLinesDevice, lenInputLinesDevice, numPatternLinesDevice, lenPatternLinesDevice, numMatchesArrDevice, allMatchLocationsDevice);
    cout << "after kernel exec (not necessarily done)" << endl;
    cudaDeviceSynchronize();
    cout << "after sync" << endl;
    cudaError_t err = cudaGetLastError();
    cout << "got last err" << endl;
    cout << err << endl;
    // copy the results to host memory
    int numMatches;
    cout << " about to memcpy" << endl;
    cudaMemcpy(&numMatches, numMatchesArrDevice, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "num mathces is " << numMatches << endl;

    matchLocation* allMatchLocations = new matchLocation[numMatches];
    cudaMemcpy(allMatchLocations, allMatchLocationsDevice, numMatches * sizeof(matchLocation), cudaMemcpyDeviceToHost);


    // prep the output file 
    ofstream outputFile("output.txt");

    // compare all the matches for lines in a pattern file to find full matches
    int sizeOfCoordArr = 10; // 10 for now... then dynamically increase if needed
    matchCoordinate* coordArr = new matchCoordinate[sizeOfCoordArr];
    int numCoords = 0;
    for (int i = 0; i < numMatches; i++) {
        matchCoordinate coor = searchForRealMatches(allMatchLocations[i], allMatchLocations, numMatches, numPatternLines, totalNumThreads);
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
    outputFile.close();
    // cleanup memory 
    for (int i = 0; i < numInputLines; i++) {
//        cudaFree(inputLinesDevice[i]);
        delete[] inputLines[i];
    }
    delete[] inputLines;
    cudaFree(inputLinesDevice);
    for (int i = 0; i < numPatternLines; i++) {
 //       cudaFree(patternLinesDevice[i]);
        delete[] patternLines[i];
    }
    delete[] patternLines;
    cudaFree(patternLinesDevice);
  //  for (int i = 0; i < totalNumThreads; i++) {
   //     cudaFree(allMatchLocationsDevice[i]);
   ///     delete[] allMatchLocations[i];
   // }
    delete[] allMatchLocations;
    cudaFree(allMatchLocationsDevice);
   // delete[] numMatchesArr;
    cudaFree(numMatchesArrDevice);
    delete[] coordArr;

    return 0;
}
