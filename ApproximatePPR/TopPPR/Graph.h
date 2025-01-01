#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <iostream>
#include <fstream>
using namespace std;

	

class Graph
{
public:
    int n;                          // # of nodes
    long m;                          // # of edges
	int** inAdjList;
	int** outAdjList;
	int* indegree;
	int* outdegree;
    Graph()
    {
        n = m = 0;
    }

    ~Graph()
    {
    	for(int i = 0; i < n ;i++){
    		delete[] inAdjList[i];
    		delete[] outAdjList[i];
    	}
        delete [] outAdjList;
        delete [] inAdjList;
		delete [] indegree;
		delete [] outdegree;
	}

    void inputGraph(string filename)
    {
    	m =0 ;
    	ifstream infile(filename.c_str());
    	infile >> n;   
    	cout << "n= " << n << endl;
    	
		indegree = new int[n];
		outdegree = new int[n];
		for(int i = 0; i < n; i++){
			indegree[i] = 0;
			outdegree[i] = 0;
		}
		int fromNode, toNode;
        int edgeCount = 0;
        while(infile >> fromNode >> toNode){
        	//infile >> fromNode >> toNode;
        	outdegree[fromNode]++;
        	indegree[toNode]++;
        }
		// while(infile >> fromNode >> toNode){
        // 	//infile >> fromNode >> toNode;
        // 	outdegree[toNode]++;
        // 	indegree[fromNode]++;
			
        // }
        cout << "..." << endl;
        inAdjList = new int*[n];
        outAdjList = new int*[n];
        int* pointer_in = new int[n];
        int* pointer_out = new int[n];
        for(int i = 0; i < n; i++){
            pointer_out[i] = 0;
            pointer_in[i] = 0;
        }
        for(int i =0; i < n; i++){
            /*if(outdegree[i] == 0){
                outdegree[i] = 1;
                outAdjList[i] = new int[1];
                outAdjList[i][0] = i; 
                inAdjList[i] = new int[indegree[i] + 1];
                inAdjList[i][0] = i;
                pointer_out[i]++;
                pointer_in[i]++;
            }*/
            //else{
        	   inAdjList[i] = new int[indegree[i]];
        	   outAdjList[i] = new int[outdegree[i]];
            //}
        }
        
        infile.clear();
        infile.seekg(0);

        clock_t t0 = clock();
        infile >> n;
        cout << "n=: " << n << endl;
        while(infile >> fromNode >> toNode){
        	//infile >> fromNode >> toNode;
        	outAdjList[fromNode][pointer_out[fromNode]++] = toNode;
        	inAdjList[toNode][pointer_in[toNode]++] = fromNode;
        	m++;
        }
        infile.close();
        clock_t t1 = clock();
        cout << "m = :" << m << endl;
        cout << "read file time: " << (t1 - t0) / (double) CLOCKS_PER_SEC << endl;
        delete[] pointer_in;
        delete[] pointer_out;
		
    }
	// void inputGraph(string filename)
	// {
	// 	m = 0;
	// 	ifstream infile(filename.c_str());
	// 	infile >> n;   
	// 	cout << "n= " << n << endl;
		
	// 	indegree = new int[n];
	// 	outdegree = new int[n];
	// 	for(int i = 0; i < n; i++){
	// 		indegree[i] = 0;
	// 		outdegree[i] = 0;
	// 	}

	// 	int fromNode, toNode;

	// 	// First pass: Count degrees
	// 	while(infile >> fromNode >> toNode){
	// 		outdegree[fromNode]++;
	// 		outdegree[toNode]++;  // Add reverse edge
	// 		indegree[toNode]++;
	// 		indegree[fromNode]++; // Add reverse edge
	// 	}

	// 	cout << "..." << endl;
	// 	inAdjList = new int*[n];
	// 	outAdjList = new int*[n];
	// 	int* pointer_in = new int[n];
	// 	int* pointer_out = new int[n];
	// 	for(int i = 0; i < n; i++){
	// 		pointer_out[i] = 0;
	// 		pointer_in[i] = 0;
	// 	}

	// 	for(int i = 0; i < n; i++){
	// 		inAdjList[i] = new int[indegree[i]];
	// 		outAdjList[i] = new int[outdegree[i]];
	// 	}
		
	// 	infile.clear();
	// 	infile.seekg(0);

	// 	clock_t t0 = clock();
	// 	infile >> n;
	// 	cout << "n=: " << n << endl;

	// 	// Second pass: Fill adjacency lists
	// 	while(infile >> fromNode >> toNode){
	// 		outAdjList[fromNode][pointer_out[fromNode]++] = toNode;
	// 		outAdjList[toNode][pointer_out[toNode]++] = fromNode;  // Add reverse edge
	// 		inAdjList[toNode][pointer_in[toNode]++] = fromNode;
	// 		inAdjList[fromNode][pointer_in[fromNode]++] = toNode;  // Add reverse edge
	// 		m += 2; // Count both edges
	// 	}

	// 	infile.close();
	// 	clock_t t1 = clock();
	// 	cout << "m = :" << m << endl;
	// 	cout << "read file time: " << (t1 - t0) / (double) CLOCKS_PER_SEC << endl;

	// 	delete[] pointer_in;
	// 	delete[] pointer_out;
	// }


	int getInSize(int vert){
		/*if(vert == 0)
			return inCount[0];
		else
			return inCount[vert] - inCount[vert - 1];*/
		return indegree[vert];
	}
	int getInVert(int vert, int pos){
		/*if(vert == 0)
			return inEdge[pos];
		else
			return inEdge[inCount[vert-1] + pos];*/
		return inAdjList[vert][pos];
	}
	int getOutSize(int vert){
		/*if(vert == 0)
			return outCount[0];
		else
			return outCount[vert] - outCount[vert - 1];*/
		return outdegree[vert];
	}
	int getOutVert(int vert, int pos){
		/*if(vert == 0)
			return outEdge[pos];
		else
			return outEdge[outCount[vert-1] + pos];*/
		return outAdjList[vert][pos];
	}
	void toFile(string filename){
		ofstream output(filename);
		for(int i = 0; i < n; i++){
			for(int j = 0; j < outdegree[i]; j++){
				output << i << " " << outAdjList[i][j] << "\n";
			}
		}
		output.close();
	}
};


#endif



// #ifndef GRAPH_H
// #define GRAPH_H

// #include <vector>
// #include <cstdlib>
// #include <algorithm>
// #include <functional>
// #include <cstdio>
// #include <iostream>
// #include <fstream>
// #include <unordered_set>
// using namespace std;

// class Graph {
// public:
//     int n;                  // Number of nodes
//     long m;                 // Number of edges (undirected, so each edge counted once)
//     int** inAdjList;        // Incoming adjacency list
//     int** outAdjList;       // Outgoing adjacency list
//     int* indegree;          // Indegree of each node
//     int* outdegree;         // Outdegree of each node

//     Graph() {
//         n = m = 0;
//         inAdjList = nullptr;
//         outAdjList = nullptr;
//         indegree = nullptr;
//         outdegree = nullptr;
//     }

//     ~Graph() {
//         for (int i = 0; i < n; i++) {
//             delete[] inAdjList[i];
//             delete[] outAdjList[i];
//         }
//         delete[] outAdjList;
//         delete[] inAdjList;
//         delete[] indegree;
//         delete[] outdegree;
//     }

//     void inputGraph(string filename) {
//         m = 0;
//         ifstream infile(filename.c_str());
//         infile >> n;
//         cout << "n= " << n << endl;

//         indegree = new int[n];
//         outdegree = new int[n];
//         for (int i = 0; i < n; i++) {
//             indegree[i] = 0;
//             outdegree[i] = 0;
//         }

//         int fromNode, toNode;
//         unordered_set<string> edgeSet; // To store edges and avoid duplicates

//         // First pass: Count degrees and handle unique edges
//         while (infile >> fromNode >> toNode) {
//             if (fromNode == toNode) continue; // Ignore self-loops

//             // Create a unique edge identifier
//             string edgeKey = fromNode < toNode
//                                  ? to_string(fromNode) + "," + to_string(toNode)
//                                  : to_string(toNode) + "," + to_string(fromNode);

//             if (edgeSet.find(edgeKey) == edgeSet.end()) {
//                 edgeSet.insert(edgeKey);
//                 outdegree[fromNode]++;
//                 indegree[toNode]++;
//                 outdegree[toNode]++;
//                 indegree[fromNode]++;
//                 m++; // Count undirected edges only once
//             }
//         }

//         cout << "..." << endl;
//         inAdjList = new int*[n];
//         outAdjList = new int*[n];
//         int* pointer_in = new int[n];
//         int* pointer_out = new int[n];
//         for (int i = 0; i < n; i++) {
//             pointer_out[i] = 0;
//             pointer_in[i] = 0;
//             inAdjList[i] = new int[indegree[i]];
//             outAdjList[i] = new int[outdegree[i]];
//         }

//         infile.clear();
//         infile.seekg(0);

//         // Second pass: Fill adjacency lists
//         infile >> n;
//         while (infile >> fromNode >> toNode) {
//             if (fromNode == toNode) continue; // Ignore self-loops

//             string edgeKey = fromNode < toNode
//                                  ? to_string(fromNode) + "," + to_string(toNode)
//                                  : to_string(toNode) + "," + to_string(fromNode);

//             if (edgeSet.find(edgeKey) != edgeSet.end()) {
//                 // Add the edge in both directions
//                 outAdjList[fromNode][pointer_out[fromNode]++] = toNode;
//                 inAdjList[toNode][pointer_in[toNode]++] = fromNode;
//                 outAdjList[toNode][pointer_out[toNode]++] = fromNode;
//                 inAdjList[fromNode][pointer_in[fromNode]++] = toNode;
//                 edgeSet.erase(edgeKey); // Mark edge as processed
//             }
//         }

//         infile.close();
//         cout << "m = :" << m << endl;
//     }

//     int getInSize(int vert) {
//         return indegree[vert];
//     }

//     int getInVert(int vert, int pos) {
//         return inAdjList[vert][pos];
//     }

//     int getOutSize(int vert) {
//         return outdegree[vert];
//     }

//     int getOutVert(int vert, int pos) {
//         return outAdjList[vert][pos];
//     }

//     void toFile(string filename) {
//         ofstream output(filename);
//         for (int i = 0; i < n; i++) {
//             for (int j = 0; j < outdegree[i]; j++) {
//                 if (i < outAdjList[i][j]) { // Avoid duplicate edges
//                     output << i << " " << outAdjList[i][j] << "\n";
//                 }
//             }
//         }
//         output.close();
//     }
// };

// #endif
