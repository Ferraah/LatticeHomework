
#include <iostream>
#include <stack>
#include <vector>
#include <cstring>
#include <utility>
#include <chrono>
#include <algorithm>
#include "parser.hpp"
#include "bool_matrix.hpp"
#include "bool_matrix_gpu.hpp"
#include "gpuAssert.hpp"



bool ENABLE_LOGGING = true;

// N-Queens node
struct Node {

    int depth; // depth in the tree

    // Bool matrix for the domains of each variable
    BoolMatrixGPU d_domains;
    //bool *d_row_is_singleton = nullptr;

    // Initialize the domains for each variable
    Node(size_t N, int *u): depth(0), d_domains(BoolMatrixGPU(N, u)) {}

    Node(const Node&) = default;
    Node(Node&&) = default;
    Node() = default;
};

void log_info(const std::string &message) {
    if (ENABLE_LOGGING) {
        std::cout << "[INFO] " << message << std::endl;
    }
}

void print_domains(BoolMatrixGPU d_domains){

    if(!ENABLE_LOGGING)
        return;


    size_t rows = d_domains.rows;
    size_t *cols = new size_t[rows];
    bool *data = new bool[d_domains.size];
    _(cudaMemcpy(cols, d_domains.cols, rows*sizeof(size_t), cudaMemcpyDeviceToHost));
    _(cudaMemcpy(data, d_domains.data, d_domains.size*sizeof(bool), cudaMemcpyDeviceToHost));
    BoolMatrix test_matrix(rows, cols);
    test_matrix.data = data;

    assert(test_matrix.data != nullptr);

    for(int i = 0; i < test_matrix.rows; i++){
        std::cout << "Domain of variable " << i << ": ";
        for(int j = 0; j < test_matrix.cols[i]; j++){
            std::cout << test_matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }

}

__global__ void find_true_index_kernel(BoolMatrixGPU domain, size_t *d_true_indices, bool *d_row_is_singleton, size_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= n)
        return;

    assert(domain.data != nullptr);
    assert(domain.cols != nullptr);

    size_t row_len = domain.cols[i];

    int count = 0;
    for(int j = 0; j < row_len; j++){
        if(domain.data[row_len+j] == true){
            d_true_indices[i] = j;
            count ++;
        }
    }

    d_row_is_singleton[i] = count == 1;
    
}

__global__ void update_domains_kernel(bool *data, size_t rows, size_t *d_offsets, int *d_C, bool *d_updated, size_t *d_last_true_row_indices, bool *d_row_is_singleton){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    size_t n = rows;
    if(i >= n*n)
        return;

    int row_a = i / n;
    int row_b = i % n;
    
    if(d_C[row_a*n + row_b] == 1){

        // Find the column index of the unique true value in the row
        if(!d_row_is_singleton[row_a])
            return;

        size_t c = d_last_true_row_indices[row_a];
        // If there is a constraint between the variables
        // Set that column in row_b to false
        if(data[d_offsets[row_b] + c] == true){
            data[d_offsets[row_b] +c] = false;
            *d_updated = true;
        }
    }

    

}


/**
 * @brief Update the domains given the constraints
 * @reture true if the domains were updated
 */
bool update_domains(BoolMatrixGPU &d_domains, int *d_C){

    bool updated = false;
    size_t rows = d_domains.rows;
    //size_t cols = d_domains.cols;

    //-------------------- FIND THE SINGLETONS --------------------  

    // Last true value index for each row
    size_t *d_last_true_row_indices; 
    // Indicates if the row has only one true value
    bool *d_row_is_singleton;

    // Allocate memory for the arrays
    _(cudaMalloc(&d_last_true_row_indices, rows*sizeof(size_t)));
    _(cudaMalloc(&d_row_is_singleton, rows*sizeof(bool)));

    // Spawn a thread per row, i.e. variable domain
    size_t total_threads = rows;
    size_t threads_per_block = 1024;
    size_t num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;


    // For each row, find if they are singletons, and the index of the true value 
    find_true_index_kernel<<<num_blocks, threads_per_block>>>(d_domains, d_last_true_row_indices, d_row_is_singleton, rows);
    _(cudaPeekAtLastError());
    _(cudaDeviceSynchronize());

    // -----------------------------------------------------------

    // -------------------- UPDATE DOMAINS ------------------------

    // Spawning a thread for each constraint, ie. n^2 threads
    total_threads = rows*rows;
    threads_per_block = 1024;
    num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    // Flag to indicate if the domains were updated
    bool *d_updated;
    _(cudaMalloc(&d_updated, sizeof(bool))); 
    _(cudaMemset(d_updated, false, sizeof(bool)));
    

    // Update the domain following the constraints
    update_domains_kernel<<<num_blocks, threads_per_block>>>(d_domains.data, d_domains.rows, d_domains.d_offsets, d_C, d_updated, d_last_true_row_indices, d_row_is_singleton);
    _(cudaPeekAtLastError());
    _(cudaDeviceSynchronize());

    // Copy the updated flag to the host
    _(cudaMemcpy(&updated, d_updated, sizeof(bool), cudaMemcpyDeviceToHost));
    // -----------------------------------------------------------
   
    // Free memory
    cudaFree(d_updated);
    cudaFree(d_last_true_row_indices);
    cudaFree( d_row_is_singleton);

    return updated;
}


/**
 * @brief Generate children nodes by branching on the variable_i
 * @param parent: Parent node
 * @param variable_i: Variable to branch on
 * @param C: Constraints matrix
 * @return Children nodes
 */
std::vector<Node> generate_children(const Node& parent, int variable_i, int **C){

    // Children nodes generated by branching on the variable_i
    std::vector<Node> children;
   
    // Get Boolean matrix GPU from the parent node
    BoolMatrixGPU d = parent.d_domains;

    // Copy the row of the variable to branch on, on the host
    size_t row_len;
    _(cudaMemcpy(&row_len, d.cols + variable_i, sizeof(size_t), cudaMemcpyDeviceToHost));

    bool *i_row = new bool[row_len];
    _(cudaMemcpy(i_row, d[variable_i], row_len*sizeof(bool), cudaMemcpyDeviceToHost));

    // For each true value in the domain of the variable, create a child node
    for(int j = 0; j < row_len; j++){
        if(i_row[j] == true){

            // New child node
            Node child(parent);
            child.depth  = parent.depth + 1;
         
            // Set the domain of the variable to be all zero apart from the unique valuec
            _(cudaMemset(child.d_domains[variable_i], false, row_len));
            _(cudaMemset(child.d_domains[variable_i] + j, true, 1));

            children.push_back(std::move(child));
        }
    }

    return children;
}

/**
 * @brief Evaluate the node and branch if solution is not found
 * @param parent: Parent node
 * @param stack: Stack of other nodes
 * @param C: Constraints matrix
 * @param tree_loc: Number of nodes explored so far
 * @param num_sol: Number of solutions found so far
 */
void evaluate_and_branch(Node &parent, std::stack<Node>& stack, int *d_C, int **C, size_t& tree_loc, size_t& num_sol){

    // Copy node so we can update the domains
    int curr_depth = parent.depth;
    int rows = parent.d_domains.rows; 
    //int cols = parent.d_domains.cols;

    log_info("Evaluating node at depth " + std::to_string(curr_depth));

    // If we are at the last variable and the domain has only one value,
    // because we have branched duing last iteration. It means that this is a solution.
    if(curr_depth == rows){
        log_info("Solution found!");
        num_sol++;
        return;
    }


    //-------------------- FIXPOINT -------------------- 

    bool updated = false;
    do{
        // Fixpoint: Update the domains until we can't update them anymore
        log_info("Updating domains...");
        updated = update_domains(parent.d_domains, d_C);
        print_domains(parent.d_domains);

        // Continue to update the domains until no restriction can be applied
    }while(updated);

    log_info("Updated domains: ");
    print_domains(parent.d_domains);

    // ------------------------------------------------

    // To continue, we necessarily need to branch 

    // Generate branches from current variable domain (curr_depth is also current variable index)
    std::vector<Node> children = generate_children(parent, curr_depth, C);

    log_info("Number of children: " + std::to_string(children.size()));

    for(int i = 0; i < children.size(); i++){
        log_info("Pushing child " + std::to_string(i) + " with domain: ");
        print_domains(children[i].d_domains);
        stack.push(std::move(children[i]));
        tree_loc++;
    }
    // ------------------------------------------------
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
    int *d_C;
    
    _(cudaMalloc(&d_C, n * n * sizeof(int))); 
    for (size_t i = 0; i < n; i++)
        _(cudaMemcpy(d_C + i * n, C[i], n * sizeof(int), cudaMemcpyHostToDevice)); 
    

    // Nodes tree
    std::stack<Node> stack;

    Node root(n, u);

    stack.push(std::move(root));

    assert(root.d_domains.data != nullptr);
    print_domains(stack.top().d_domains);

    // Counters
    size_t explored_nodes = 0;
    size_t explored_sol = 0;

    // Domains for each variable
    while(stack.size() != 0){
        Node current(std::move(stack.top()));
        stack.pop();
        evaluate_and_branch(current, stack, d_C, C, explored_nodes, explored_sol);
    }

    std::cout << "Number of solutions: " << explored_sol << std::endl;
    std::cout << "Number of nodes explored: " << explored_nodes << std::endl;

    return 0;
}
