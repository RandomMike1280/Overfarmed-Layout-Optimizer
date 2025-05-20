#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

enum { ITEM_NONE = 0, ITEM_PLANTER = 1, ITEM_FERTILIZER = 2, ITEM_SPRINKLER = 3, ITEM_HARVESTER = 4 };

// global config
static int rows, cols, pop_size, top_k, eval_ticks, grow_time, generations_count;
static double mutation_rate, cross_rate;
static double ranges_arr[4], costs_arr[4];

// Pre-computed range values to avoid repeated calculations
static int range_sides[4];
static int range_halves[4];

// population and fitness arrays
static int **population_arr;
static double *fitness_arr;
static int *best_grid_global;
static double best_fitness_global;

// Reusable simulation buffers to avoid repeated allocations
static int *sim_grid_buffer = NULL;
static double *progress_buffer = NULL;
static int *time_req_buffer = NULL;
static char *fertilized_buffer = NULL;
static char *watered_buffer = NULL;
static int sim_rows, sim_cols;

// forward declarations for utilities used in gene pool
static int *allocate_grid();
static void copy_grid(int *dest, int *src);

// Memory pool for grid allocations
#define MEMORY_POOL_SIZE 1000
static int **grid_memory_pool = NULL;
static int grid_memory_pool_index = 0;

static void initialize_memory_pool() {
    grid_memory_pool = (int**)malloc(MEMORY_POOL_SIZE * sizeof(int*));
    for (int i = 0; i < MEMORY_POOL_SIZE; i++) {
        grid_memory_pool[i] = (int*)malloc(rows * cols * sizeof(int));
    }
    grid_memory_pool_index = 0;
}

static void free_memory_pool() {
    if (grid_memory_pool) {
        for (int i = 0; i < MEMORY_POOL_SIZE; i++) {
            free(grid_memory_pool[i]);
        }
        free(grid_memory_pool);
        grid_memory_pool = NULL;
    }
}

static int* get_grid_from_pool() {
    if (!grid_memory_pool) {
        initialize_memory_pool();
    }
    if (grid_memory_pool_index >= MEMORY_POOL_SIZE) {
        grid_memory_pool_index = 0; // Reuse from the beginning
    }
    return grid_memory_pool[grid_memory_pool_index++];
}

// Hash table for gene pool lookups
#define HASH_TABLE_SIZE 10007 // Prime number for better distribution
typedef struct GeneNode {
    int *grid;
    double fitness;
    struct GeneNode *next;
} GeneNode;

static GeneNode **gene_hash_table = NULL;
static int gene_count = 0;

static unsigned int hash_grid(int *grid) {
    unsigned int hash = 0;
    for (int i = 0; i < rows * cols; i++) {
        hash = hash * 31 + grid[i];
    }
    return hash % HASH_TABLE_SIZE;
}

static void initialize_gene_hash_table() {
    gene_hash_table = (GeneNode**)calloc(HASH_TABLE_SIZE, sizeof(GeneNode*));
    gene_count = 0;
}

static int is_known_gene(int *grid) {
    if (!gene_hash_table) return 0;
    
    unsigned int hash = hash_grid(grid);
    GeneNode *node = gene_hash_table[hash];
    
    while (node) {
        if (memcmp(node->grid, grid, rows * cols * sizeof(int)) == 0) {
            return 1;
        }
        node = node->next;
    }
    return 0;
}

static void add_gene_entry(int *grid, double fitness) {
    if (!gene_hash_table) {
        initialize_gene_hash_table();
    }
    
    if (is_known_gene(grid)) return; // Already in hash table
    
    unsigned int hash = hash_grid(grid);
    GeneNode *new_node = (GeneNode*)malloc(sizeof(GeneNode));
    new_node->grid = get_grid_from_pool();
    copy_grid(new_node->grid, grid);
    new_node->fitness = fitness;
    new_node->next = gene_hash_table[hash];
    gene_hash_table[hash] = new_node;
    gene_count++;
}

static void purge_gene_pool() {
    if (!gene_hash_table || gene_count < HASH_TABLE_SIZE) return;
    
    double threshold = best_fitness_global * 0.5;
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        GeneNode **prev_ptr = &gene_hash_table[i];
        GeneNode *node = gene_hash_table[i];
        
        while (node) {
            if (node->fitness < threshold) {
                *prev_ptr = node->next;
                free(node);
                node = *prev_ptr;
                gene_count--;
            } else {
                prev_ptr = &node->next;
                node = node->next;
            }
        }
    }
}

static void free_gene_hash_table() {
    if (!gene_hash_table) return;
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        GeneNode *node = gene_hash_table[i];
        while (node) {
            GeneNode *next = node->next;
            free(node);
            node = next;
        }
    }
    free(gene_hash_table);
    gene_hash_table = NULL;
    gene_count = 0;
}

// utilities
static inline double rand_double() { return rand() / (double)RAND_MAX; }
static inline int rand_int(int max) { return rand() % max; }
static int *allocate_grid() { return (int *)malloc(rows * cols * sizeof(int)); }
static void copy_grid(int *dest, int *src) { memcpy(dest, src, rows * cols * sizeof(int)); }

// initialize optimizer
DLL_EXPORT void init_optimizer(int _rows, int _cols, int _generations,
    int _pop_size, int _top_k, int _eval_ticks, int _grow_time,
    double _mutation_rate, double _cross_rate,
    double _ranges[4], double _costs[4]) {
    rows = _rows; cols = _cols; generations_count = _generations;
    pop_size = _pop_size; top_k = _top_k; eval_ticks = _eval_ticks; grow_time = _grow_time;
    mutation_rate = _mutation_rate; cross_rate = _cross_rate;
    
    // Pre-compute range values
    for (int i = 0; i < 4; i++) {
        ranges_arr[i] = _ranges[i];
        costs_arr[i] = _costs[i];
        range_sides[i] = (int)_ranges[i];
        range_halves[i] = range_sides[i] / 2;
    }
    
    // Initialize simulation buffers
    sim_rows = rows * 3;
    sim_cols = cols * 3;
    sim_grid_buffer = (int*)malloc(sim_rows * sim_cols * sizeof(int));
    progress_buffer = (double*)malloc(sim_rows * sim_cols * sizeof(double));
    time_req_buffer = (int*)malloc(sim_rows * sim_cols * sizeof(int));
    fertilized_buffer = (char*)malloc(sim_rows * sim_cols);
    watered_buffer = (char*)malloc(sim_rows * sim_cols);
    
    // seed RNG
    srand((unsigned)time(NULL));
    
    // Initialize memory pool
    initialize_memory_pool();
    
    // allocate population
    population_arr = (int**)malloc(pop_size * sizeof(int *));
    fitness_arr = (double*)malloc(pop_size * sizeof(double));
    
    // initialize first generation: 80% random, 20% vertical-axis symmetric
    int sym_count = pop_size / 5;
    int rand_count = pop_size - sym_count;
    for (int i = 0; i < pop_size; i++) {
        population_arr[i] = get_grid_from_pool();
        if (i < rand_count) {
            // fully random layout
            for (int j = 0; j < rows * cols; j++) {
                population_arr[i][j] = rand_int(5);
            }
        } else {
            // symmetric layout along vertical axis
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < (cols + 1) / 2; c++) {
                    int val = rand_int(5);
                    population_arr[i][r*cols + c] = val;
                    population_arr[i][r*cols + (cols - 1 - c)] = val;
                }
            }
        }
    }
    best_grid_global = allocate_grid();
    best_fitness_global = -1e308;
    
    // initialize gene pool
    initialize_gene_hash_table();
}

// simulate one layout
static double simulate_grid(int *grid) {
    // Reset buffers instead of reallocating
    memset(progress_buffer, 0, sim_rows * sim_cols * sizeof(double));
    memset(time_req_buffer, 0, sim_rows * sim_cols * sizeof(int));
    memset(fertilized_buffer, 0, sim_rows * sim_cols);
    memset(watered_buffer, 0, sim_rows * sim_cols);
    
    // tile base grid into 3x3 layout
    for (int ti = 0; ti < 3; ti++) {
        for (int tj = 0; tj < 3; tj++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    sim_grid_buffer[(ti*rows + r) * sim_cols + (tj*cols + c)] = grid[r*cols + c];
                }
            }
        }
    }
    
    double harvested = 0.0;
    double cost_penalty = 0.0;
    
    // Pre-calculate placement cost to avoid repeated calculations
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int t = grid[r*cols + c];
            if (t > ITEM_NONE) {
                cost_penalty += costs_arr[t-1] * 9; // Multiply by 9 for the 3x3 tiling
            }
        }
    }
    
    // Pre-compute planter and harvester coverage maps to avoid repeated range checks
    char *planter_coverage = (char*)calloc(sim_rows * sim_cols, 1);
    char *harvester_coverage = (char*)calloc(sim_rows * sim_cols, 1);
    
    for (int r = 0; r < sim_rows; r++) {
        for (int c = 0; c < sim_cols; c++) {
            int t = sim_grid_buffer[r*sim_cols + c];
            
            if (t == ITEM_PLANTER) {
                int half = range_halves[0];
                int side = range_sides[0];
                for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++) {
                    for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                        planter_coverage[i*sim_cols + j] = 1;
                    }
                }
            }
            else if (t == ITEM_HARVESTER) {
                int half = range_halves[3];
                int side = range_sides[3];
                for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++) {
                    for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                        harvester_coverage[i*sim_cols + j] = 1;
                    }
                }
            }
        }
    }
    
    // Count unreachable squares for penalty calculation
    int unreach_planter = 0, unreach_harv = 0;
    for (int i = 0; i < sim_rows * sim_cols; i++) {
        if (!planter_coverage[i]) unreach_planter++;
        if (!harvester_coverage[i]) unreach_harv++;
    }
    
    // Simulation loop
    for (int tick = 0; tick < eval_ticks; tick++) {
        // planter - use pre-computed coverage map
        for (int i = 0; i < sim_rows * sim_cols; i++) {
            if (planter_coverage[i] && sim_grid_buffer[i] == ITEM_NONE && time_req_buffer[i] == 0) {
                time_req_buffer[i] = grow_time;
            }
        }
        
        // fertilizer and sprinkler
        for (int r = 0; r < sim_rows; r++) {
            for (int c = 0; c < sim_cols; c++) {
                int idx = r * sim_cols + c;
                int t = sim_grid_buffer[idx];
                
                if (t == ITEM_FERTILIZER) {
                    int half = range_halves[1];
                    int side = range_sides[1];
                    for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++) {
                        for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                            int target_idx = i*sim_cols + j;
                            if (time_req_buffer[target_idx] > 0 && !fertilized_buffer[target_idx]) {
                                fertilized_buffer[target_idx] = 1;
                                if (watered_buffer[target_idx]) {
                                    time_req_buffer[target_idx] = (int)(grow_time * 0.5);
                                } else {
                                    time_req_buffer[target_idx] = (int)(grow_time * 0.75);
                                }
                            }
                        }
                    }
                }
                else if (t == ITEM_SPRINKLER) {
                    int half = range_halves[2];
                    int side = range_sides[2];
                    for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++) {
                        for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                            int target_idx = i*sim_cols + j;
                            if (time_req_buffer[target_idx] > 0 && !watered_buffer[target_idx]) {
                                watered_buffer[target_idx] = 1;
                                if (fertilized_buffer[target_idx]) {
                                    time_req_buffer[target_idx] = (int)(grow_time * 0.5);
                                } else {
                                    time_req_buffer[target_idx] = (int)(grow_time * 0.75);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // growth - process in a single loop
        for (int i = 0; i < sim_rows*sim_cols; i++) {
            if (time_req_buffer[i] > 0) {
                progress_buffer[i] += 1.0 / time_req_buffer[i];
                
                // Inline harvester check to avoid another loop
                if (progress_buffer[i] >= 1.0 && harvester_coverage[i]) {
                    harvested += 1.0;
                    progress_buffer[i] = 0.0;
                    time_req_buffer[i] = 0;
                    fertilized_buffer[i] = 0;
                    watered_buffer[i] = 0;
                }
            }
        }
    }
    
    // Calculate final score
    double penalty_factor = 1.0 - 0.05 * unreach_planter - 0.05 * unreach_harv;
    if (penalty_factor < 0) penalty_factor = 0;
    double net_score = harvested - cost_penalty;
    double final_score = net_score * penalty_factor;
    
    free(planter_coverage);
    free(harvester_coverage);
    
    return final_score;
}

// Optimized selection function
static void select_parents(int *parent_indices, int parent_count) {
    // Use tournament selection instead of complex Gumbel-top-k
    // This maintains diversity while being much faster
    
    for (int i = 0; i < parent_count; i++) {
        // Select 3 random individuals for tournament
        int idx1 = rand_int(pop_size);
        int idx2 = rand_int(pop_size);
        int idx3 = rand_int(pop_size);
        
        // Find the best one
        int best_idx = idx1;
        if (fitness_arr[idx2] > fitness_arr[best_idx]) best_idx = idx2;
        if (fitness_arr[idx3] > fitness_arr[best_idx]) best_idx = idx3;
        
        parent_indices[i] = best_idx;
    }
}

// run GA
DLL_EXPORT void run_optimizer(int *out_best_grid, double *out_best_fitness) {
    int stagnant_count = 0;
    double prev_best = best_fitness_global;
    
    // Pre-allocate parent indices array
    int parent_count = pop_size / 2;
    int *parent_indices = (int*)malloc(parent_count * sizeof(int));
    
    // Pre-allocate new population array
    int **newpop = (int**)malloc(pop_size * sizeof(int*));
    
    for (int gen = 0; gen < generations_count; gen++) {
        // evaluate fitness
        #pragma omp parallel for if(pop_size > 100)
        for (int i = 0; i < pop_size; i++) {
            fitness_arr[i] = simulate_grid(population_arr[i]);
        }
        
        // update gene pool
        for (int i = 0; i < pop_size; i++) {
            add_gene_entry(population_arr[i], fitness_arr[i]);
        }
        
        // Periodically purge gene pool to prevent excessive memory usage
        if (gen % 10 == 0) {
            purge_gene_pool();
        }
        
        // select parents using optimized selection
        select_parents(parent_indices, parent_count);
        
        // update best individual
        int best_i = 0;
        for (int i = 1; i < pop_size; i++) {
            if (fitness_arr[i] > fitness_arr[best_i]) best_i = i;
        }
        
        if (fitness_arr[best_i] > best_fitness_global) {
            best_fitness_global = fitness_arr[best_i];
            copy_grid(best_grid_global, population_arr[best_i]);
        }
        
        // early stopping after 50 gens without improvement
        if (best_fitness_global > prev_best) {
            stagnant_count = 0;
            prev_best = best_fitness_global;
        } else {
            stagnant_count++;
        }
        
        if (stagnant_
(Content truncated due to size limit. Use line ranges to read in chunks)