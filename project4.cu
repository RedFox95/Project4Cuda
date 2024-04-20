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

matchCoordinate searchForRealMatches(matchLocation match, matchLocation** allMatches, int* numMatchesArr, int numPatternLines, int totalNumThreads) {
    //cout << "-> searchForRealMAtches x:" << match.x << " y: " << match.y << " pl: " << match.pl << endl;
    matchLocation** patternMatchLocations = new matchLocation * [numPatternLines];
    for (int i = 0; i < numPatternLines; i++) patternMatchLocations[i] = nullptr;

    patternMatchLocations[match.pl] = &match;
    for (int i = 0; i < totalNumThreads; i++) {
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

__device__ void findSubStr(char*str, int row, int strLen, char* subStr, int patternLineNum, int subStrLen, int pos, int*foundPos) {
    printf("-> findSubStr row %d, strLen %d, patternLineNum %d, subStrLen %d, pos %d\n", row, strLen, patternLineNum, subStrLen, pos);
    //*foundPos = -1; // initialize to -1 to assume not found // TODO this causes memory problem - maybe need some sort of malloc type thing here
    //printf("r %d: difference is %d \n", row, strLen-subStrLen);
    *foundPos = -1; // assume no match found
    for (int i = pos; i <= strLen - subStrLen; i++) {
        printf("r %d: i is %d\n", row, i);
        bool found = true;
        for (int j = 0; j < subStrLen; j++) {
            printf("r %d: j is %d and i+j is %d comparing %c vs %c\n", row, j, i + j, str[(row * strLen) + i + j], subStr[(patternLineNum * subStrLen)  + j]);
            if (str[(row*strLen) + i + j] != subStr[(patternLineNum * subStrLen)+j]) {
                printf("r %d: not a match\n", row);
                found = false;
                break;
            }
            else {
                printf("r %d: part of a match!\n", row);
            }
        }
        if (found) {
            *foundPos = i;
            printf("r %d: FOUND match in device foundPos is %d\n", row, *foundPos);
            break;
        }
    }
}

__global__ void findPartialMatches(char* inputLines, char* patternLines, int* numInputLines, int* lenInputLines, int* numPatternLines, int* lenPatternLines, int* numMatchesArr, matchLocation** allMatches, int*sizeArr) {
//    printf("HELLO FROM KERNEL!\n");
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int threadIndex = threadId * *lenInputLines;
    printf("tmg numinputlines %d, numpattern lines %d\n", *numInputLines, *numPatternLines); 
    printf("tMg leninputlines %d, lenpattern lines %d\n", *lenInputLines, *lenPatternLines); 
    printf("Tmg thread id is %d and index is %d\n", threadId, threadIndex);
    numMatchesArr[threadId] = 0; // initialize all numMatches to 0
    sizeArr[threadId] = 10; // initialize all array sizes to 10
    printf("tId %d: nummatches set to %d and size set to %d\n", threadId, numMatchesArr[threadId], sizeArr[threadId]);
    
    allMatches[threadId] = new matchLocation[sizeArr[threadId]];
    if (threadId < *numInputLines) {
        //printf("de did numinputlines\n");
       // int sizeOfMatchArr = 10; // 10 for now... then dynamically increase if needed
        //allMatches[threadId] = new matchLocation[sizeOfMatchArr]; // store the matches at the index for this thread
        for (int j = 0; j < *numPatternLines; j++) {
            printf("tId %d: in loop j is %d\n", threadId, j);
            int pos = 0;
            //string jPatternLine(patternLines[j], patternLines[j] + *lenPatternLines - 1);
            //string iInputLine(inputLines[threadId], inputLines[threadId] + *lenInputLines - 1);
            //int found = iInputLine.find(jPatternLine, pos);
            int found = -1;
//            findSubStr(inputLines[threadId], *lenInputLines - 1, patternLines[j], *lenPatternLines - 1, 0, found);
            printf("before calling findSubStr j is %d\n", j);
            findSubStr(inputLines, threadId, *lenInputLines, patternLines, j, *lenPatternLines, 0, &found); // idk if these -1 are needed in this implementation
            printf("tId %d: after find found is %d and j is %d\n", threadId, found, j);
            while (found != -1) {
                if (numMatchesArr[threadId] >= sizeArr[threadId]) {
                    // increase matchArr size 
                    size_t biggerSize = sizeArr[threadId] * 2;
                    matchLocation* biggerArr = new matchLocation[biggerSize];
                    memcpy(biggerArr, allMatches[threadId], sizeof(matchLocation) * numMatchesArr[threadId]);
                    delete[] allMatches[threadId];
                    allMatches[threadId] = biggerArr;
                    sizeArr[threadId] = biggerSize;
                }
                // store the match
                struct matchLocation m = { threadId, found, j }; // where threadId is the row number, found is the col number, and j is the pattern line number
                printf("tID %d: made the matchLocation\n", threadId);
                printf("tID %d: allmatches arr storing in [%d][%d]\n", threadId, threadId, numMatchesArr[threadId]);
                allMatches[threadId][numMatchesArr[threadId]] = m;
                printf("tId %d: after storing matchlocation and before incrementing num matches which is currenmtly %d\n", threadId, numMatchesArr[threadId]);
                // update pos, numMatches, and found for the next iteration
                pos = found + 1;
                numMatchesArr[threadId]++;
                printf("tId %d: after incrementing nummatchesarr to %d\n", threadId, numMatchesArr[threadId]);
  //              findSubStr(inputLines[threadId], *lenInputLines - 1, patternLines[j], *lenPatternLines - 1, pos, found);
                printf("tId %d: abvout to call findSubStr again j is %d\n", threadId, j);
                findSubStr(inputLines, threadId,  *lenInputLines, patternLines, j, *lenPatternLines, pos, &found);
            }
            printf("tId %d: after while loop\n", threadId);
        }
        printf("tId %d: after for loop\n", threadId);
    }
    printf("Thread %d num matches is %d\n", threadId, numMatchesArr[threadId]);
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
    cout << " length of 1d arr is " << numInputLines * lenInputLines  << endl;

    string line;
    int lineNum = 0; // for indexing into the allLines arr
    while (getline(file, line)) {
        strcpy_s(inputLines[lineNum], lenInputLines+1, line.c_str());
        cout << inputLines[lineNum] << endl;
        for (int i = 0; i < lenInputLines; i++) {
            cout << "inserting " << line.c_str()[i] << " at " << (lineNum * lenInputLines) + i << endl;
            inputLine1d[(lineNum * lenInputLines) + i] = line.c_str()[i];
        }
        
        lineNum++;
    }
    for (int i = 0; i < numInputLines * lenInputLines; i++) {
        cout << inputLine1d[i];
    }
    cout << "\nthat was it " << endl;
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
    cout << " length of 1d arr is " << numPatternLines * lenPatternLines << endl;

    lineNum = 0; // for indexing into the pattern arr
    while (getline(patternFileStream, line)) {
        strcpy_s(patternLines[lineNum], lenPatternLines+1, line.c_str());
        for (int i = 0; i < lenPatternLines; i++) {
            cout << "inserting " << line.c_str()[i] << " at " << (lineNum * lenPatternLines) + i << endl;
            patternLine1d[(lineNum * lenPatternLines) + i] = line.c_str()[i];
        }
        lineNum++;
    }
    for (int i = 0; i < numPatternLines * lenPatternLines; i++) {
        cout << patternLine1d[i];
    }
    cout << "\nthat was it " << endl;
    cout << "numinput lines is " << numInputLines << " and num pattern lines is " << numPatternLines << endl;
    cout << "about to allocate mem for inputLines and patternlines" << endl;
    // allocate memory on device for inputLines and patternLines and copy to the device memory
    char* inputLinesDevice;
  //  cudaMalloc((void**)&inputLinesDevice, lenInputLines * numInputLines * sizeof(char)); // inputLinesDevice will be the 2d array flattened 
    cudaMalloc((void**)&inputLinesDevice, lenInputLines * numInputLines * sizeof(char)); // inputLinesDevice will be the 2d array flattened 
    cout << "cudamalloc for inputLinesDevice done" << endl;
  //  cudaMemcpy((void*)inputLinesDevice, (void*)inputLines, lenInputLines * numInputLines * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)inputLinesDevice, (void*)inputLine1d, lenInputLines * numInputLines * sizeof(char), cudaMemcpyHostToDevice);
    cout << "cudaMemcpy for inputLinesDevice done" << endl;
  /*  for (int i = 0; i < numInputLines; i++) {
        cout << " in loo i is " << i << endl;
        cudaMalloc((void**) & (inputLinesDevice[i]), lenInputLines * sizeof(char));
        cout << " malloc not to memcpy" << endl;
        cudaMemcpy(inputLinesDevice[i], inputLines[i], lenInputLines * sizeof(char), cudaMemcpyHostToDevice);
    }*/
   // size_t inputPitch;
    //cudaMallocPitch((void**)inputLinesDevice, &inputPitch,  lenInputLines * sizeof(char), numInputLines);
    //size_t spitch = numPatternLines * lenInputLines *sizeof(char);
    //cudaMemcpy2DToArray((void**)inputLinesDevice, 0, 0, inputLines, spitch, lenPatternLines * sizeof(char), numInputLines, cudaMemcpyHostToDevice);
    cout << "malloc and memcpy done for each line of input" << endl;
    char* patternLinesDevice;
    cout << "about to malloc pattern" << endl;
//    cudaMalloc((void**)&patternLinesDevice, lenPatternLines * numPatternLines * sizeof(char));
    cudaMalloc((void**)&patternLinesDevice, lenPatternLines * numPatternLines * sizeof(char));
    cout << "after malloc 1" << endl;
//    cudaMemcpy((void*)patternLinesDevice, (void*)patternLines, lenPatternLines * numPatternLines * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)patternLinesDevice, (void*)patternLine1d, lenPatternLines * numPatternLines * sizeof(char), cudaMemcpyHostToDevice);
    cout << "after memcpy1" << endl;
   /* for (int i = 0; i < numPatternLines; i++) {
        cudaMalloc(&patternLinesDevice[i], lenPatternLines * sizeof(char));
        cudaMemcpy(patternLinesDevice[i], patternLines[i], lenPatternLines * sizeof(char), cudaMemcpyHostToDevice);
    }
    cout << "same done for pattern" << endl;
    */
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

//    int* numThreadsDevice;
//    cudaMalloc((void**)&numThreadsDevice, sizeof(int));
//    cudaMemcpy((void*)numThreadsDevice, &numThreads, sizeof(int), cudaMemcpyHostToDevice);

    // setup pointers to get the results from device memory in allMatchLocations and numMatchesArr
    matchLocation** allMatchLocationsDevice;
    cudaMalloc(&allMatchLocationsDevice, totalNumThreads * sizeof(matchLocation*));
    cout << "after malloc mathclocations device " << endl;
    int* numMatchesArrDevice;
    cudaMalloc(&numMatchesArrDevice, totalNumThreads * sizeof(int));
    int* sizeArrDevice;
    cudaMalloc(&sizeArrDevice, totalNumThreads * sizeof(int));
//    cout << "after malloc num matches device " << endl; // idk if this is needed
//    for (int i = 0; i < numBlocks; i++) {
//        cout << i << endl;
//        cudaMalloc((void**) &allMatchLocationsDevice[i], numBlocks * sizeof(int));
//    }
    cout << "about to start kernel" << endl;
    // start the kernel to find partial matches
    findPartialMatches<<<numBlocks,numThreads>>>(inputLinesDevice, patternLinesDevice, numInputLinesDevice, lenInputLinesDevice, numPatternLinesDevice, lenPatternLinesDevice, numMatchesArrDevice, allMatchLocationsDevice, sizeArrDevice);
    cout << "after kernel exec (not necessarily done)" << endl;
    cudaDeviceSynchronize();
    cout << "after sync" << endl;
    cudaError_t err = cudaGetLastError();
    cout << "got last err" << endl;
    cout << err << endl;
    // copy the results to host memory
    int* numMatchesArr = new int[totalNumThreads];
    matchLocation** allMatchLocations = new matchLocation * [totalNumThreads];

    for (int i = 0; i < totalNumThreads; i++) {
        cudaMemcpy((void*)numMatchesArr[i], (void*) & numMatchesArrDevice[i], sizeof(int), cudaMemcpyDeviceToHost);
        cout << "nummathces is " << numMatchesArr[i] << endl;
        if (numMatchesArr[i] > 0) {
            cudaMemcpy(allMatchLocations[i], allMatchLocationsDevice[i], numMatchesArr[i] * sizeof(matchLocation*), cudaMemcpyDeviceToHost);
        }
    }
    cout << "after copying mem from dev to host for nummatch which is " << numMatchesArr[1] << " and allmatchloc" << endl;

    // prep the output file 
    ofstream outputFile("output.txt");

    // compare all the matches for lines in a pattern file to find full matches
    int sizeOfCoordArr = 10; // 10 for now... then dynamically increase if needed
    matchCoordinate* coordArr = new matchCoordinate[sizeOfCoordArr];
    int numCoords = 0;
    for (int i = 0; i < numBlocks; i++) {
        for (int j = 0; j < numMatchesArr[i]; j++) {
            matchCoordinate coor = searchForRealMatches(allMatchLocations[i][j], allMatchLocations, numMatchesArr, numPatternLines, totalNumThreads);
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

