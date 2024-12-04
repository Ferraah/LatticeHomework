#include <iostream>
#include <stack>
#include <vector>
#include <cstring>
#include <utility>
#include <chrono>
#include <algorithm>
#include "parser.hpp"
#include "bool_matrix.hpp"

void print_domains(BoolMatrix &matrix){
        
    for(int i = 0; i < matrix.rows; i++){
        std::cout << "Domain of variable " << i << ": ";
        for(int j = 0; j < matrix.cols; j++){
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[]){

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    Data data;
    if (data.read_input(filename.c_str())){
        data.print_n();
        data.print_u();
        data.print_C();
    } else {
        std::cerr << "Failed to read input file: " << filename << std::endl;
        return 1;
    }

    // Number of variables
    int n = data.get_n();
    int *u = data.get_u();
    int **C = data.get_C();


    size_t size = 0; 
    size_t max_u = 0;
    for (int i = 0; i < n; i++){
        if(u[i]+1 > max_u)
            max_u = u[i]+1;
        size += u[i] + 1; 
    }

    bool *host_data = new bool[n*max_u];
    memset(host_data, 0, n*max_u*sizeof(bool));
    BoolMatrix matrix(n, max_u, host_data);

    print_domains(matrix);    



    return 0;
}
