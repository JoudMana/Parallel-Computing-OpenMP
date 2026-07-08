#include <iostream>
#include <omp.h>
#include <limits.h>
#include <algorithm>

using namespace std;

// Allocate n x n matrix
int** allocateMatrix(int n) {
    int** mat = new int* [n];
    for (int i = 0; i < n; i++) {
        mat[i] = new int[n];
    }
    return mat;
}

// Free matrix memory
void freeMatrix(int** mat, int n) {
    for (int i = 0; i < n; i++) {
        delete[] mat[i];
    }
    delete[] mat;
}

// Initialize matrix
void fillMatrix(int** mat, int n, int val) {
#pragma omp parallel for collapse(2) schedule(static) 
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = val;
        }
    }
}

// Parallel Matrix Sum
long long matrixSum(int** mat, int n) {
    long long total_sum = 0;
#pragma omp parallel for collapse(2) reduction(+:total_sum)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            total_sum += mat[i][j];
        }
    }
    return total_sum;
}

// Parallel Matrix Maximum Element
int matrixMax(int** mat, int n) {
    int global_max = INT_MIN;
#pragma omp parallel for collapse(2) reduction(max:global_max)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (mat[i][j] > global_max) {
                global_max = mat[i][j];
            }
        }
    }
    return global_max;
}

// Standard Multiplication (No Transpose)
int** matrixMult(int** A, int** B, int n) {
    int** C = allocateMatrix(n);

    // Standard Matrix Multiplication: Row of A * Column of B
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

int main() {
    int sizes[] = { 1500, 3000 };
    int threads[] = { 1, 2, 4, 8 };

    for (int s = 0; s < 2; s++) {
        int n = sizes[s];
        cout << "Processing Matrix Size: " << n << " * " << n << endl;

        int** A = allocateMatrix(n);
        int** B = allocateMatrix(n);

        fillMatrix(A, n, 1);
        fillMatrix(B, n, 2);

        int*** stored_results = new int** [4];
        bool all_equal = true;

        for (int t = 0; t < 4; t++) {
            int num_threads = threads[t];
            omp_set_num_threads(num_threads);
            cout << "    Number of Threads = " << num_threads << endl;

            double start = omp_get_wtime();
            long long sum = matrixSum(A, n);
            cout << "    Total Sum = " << sum << endl;
            int max = matrixMax(A, n);
            cout << "    The max = " << max << endl;
            stored_results[t] = matrixMult(A, B, n);
            double end = omp_get_wtime();
            cout << "    Multiplication done.\n    Total Time Without Transpose  = " << (end - start) << " s" << endl;

            // Check Equality
            if (t > 0) {
                bool current_match = true;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if (stored_results[t][i][j] != stored_results[0][i][j]) {
                            current_match = false;
                            all_equal = false;
                            i = n; j = n;
                        }
                    }
                }
                if (!current_match) cout << "    Mismatch detected!" << endl;
            }
            cout << "\n" << endl;
        }

        if (all_equal) cout << "    SUCCESS: All results match." << endl;
        else cout << "    FAILURE: Results differ." << endl;
        cout << "\n" << endl;

        freeMatrix(A, n);
        freeMatrix(B, n);

        for (int t = 0; t < 4; t++) freeMatrix(stored_results[t], n);
        delete[] stored_results;
    }
    return 0;
}