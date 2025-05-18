#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

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

// population and fitness arrays
static int **population_arr;
static double *fitness_arr;
static int *best_grid_global;
static double best_fitness_global;

// precomputed for simulation tiling
static int sim_rows, sim_cols, sim_size;
static int side_arr[4], half_arr[4];
// reusable buffers
static double *progress_arr;
static int *time_req_arr;
static char *fertilized_arr, *watered_arr;

// forward declarations for utilities used in gene pool
static int *allocate_grid();
static void copy_grid(int *dest, int *src);

// gene pool support
typedef struct HashNode { int *grid; double fitness; struct HashNode *next; } HashNode;
static HashNode **gene_table;
static int table_size;

static inline uint64_t hash_grid(int *grid) {
    uint64_t h = 146527;
    for (int i = 0, n = rows*cols; i < n; i++) h = h * 31 + (uint32_t)grid[i];
    return h;
}

static void initialize_gene_pool() {
    table_size = pop_size * 4;
    gene_table = calloc(table_size, sizeof(HashNode*));
}

static int is_known_gene(int *grid) {
    if (!gene_table) initialize_gene_pool();
    uint64_t h = hash_grid(grid);
    int idx = h % table_size;
    for (HashNode *node = gene_table[idx]; node; node = node->next) {
        if (memcmp(node->grid, grid, rows*cols*sizeof(int)) == 0) return 1;
    }
    return 0;
}

static void add_gene_entry(int *grid, double fitness) {
    if (!gene_table) initialize_gene_pool();
    uint64_t h = hash_grid(grid);
    int idx = h % table_size;
    HashNode *node = malloc(sizeof(HashNode));
    node->grid = allocate_grid();
    copy_grid(node->grid, grid);
    node->fitness = fitness;
    node->next = gene_table[idx];
    gene_table[idx] = node;
}

static void purge_gene_pool() {
    double threshold = best_fitness_global * 0.5;
    for (int i = 0; i < table_size; i++) {
        HashNode *prev = NULL, *cur = gene_table[i];
        while (cur) {
            if (cur->fitness < threshold) {
                HashNode *del = cur;
                if (prev) prev->next = cur->next;
                else gene_table[i] = cur->next;
                cur = cur->next;
                free(del->grid);
                free(del);
            } else {
                prev = cur;
                cur = cur->next;
            }
        }
    }
}

// utilities
static double rand_double() { return rand() / (double)RAND_MAX; }
static int rand_int(int max) { return rand() % max; }
static int *allocate_grid() { return (int *)malloc(rows * cols * sizeof(int)); }
static void copy_grid(int *dest, int *src) { memcpy(dest, src, rows * cols * sizeof(int)); }

#define GET(r,c) (grid[((r)%rows)*cols + ((c)%cols)])

// initialize optimizer
DLL_EXPORT void init_optimizer(int _rows, int _cols, int _generations,
    int _pop_size, int _top_k, int _eval_ticks, int _grow_time,
    double _mutation_rate, double _cross_rate,
    double _ranges[4], double _costs[4]) {
    rows = _rows; cols = _cols; generations_count = _generations;
    pop_size = _pop_size; top_k = _top_k; eval_ticks = _eval_ticks; grow_time = _grow_time;
    mutation_rate = _mutation_rate; cross_rate = _cross_rate;
    for (int i = 0; i < 4; i++) {
        ranges_arr[i] = _ranges[i]; costs_arr[i] = _costs[i];
    }
    // seed RNG
    srand((unsigned)time(NULL));
    // prepare tiled simulation parameters
    sim_rows = rows * 3;
    sim_cols = cols * 3;
    sim_size = sim_rows * sim_cols;
    // allocate reusable buffers
    progress_arr = malloc(sim_size * sizeof(double));
    time_req_arr = malloc(sim_size * sizeof(int));
    fertilized_arr = malloc(sim_size * sizeof(char));
    watered_arr = malloc(sim_size * sizeof(char));
    // precompute side and half for each tool
    for (int t = 0; t < 4; t++) {
        side_arr[t] = (int)ranges_arr[t];
        half_arr[t] = side_arr[t] / 2;
    }
    // allocate population
    population_arr = malloc(pop_size * sizeof(int *));
    fitness_arr = malloc(pop_size * sizeof(double));
    // initialize first generation: 80% random, 20% vertical-axis symmetric
    int sym_count = pop_size / 5;
    int rand_count = pop_size - sym_count;
    for (int i = 0; i < pop_size; i++) {
        population_arr[i] = allocate_grid();
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
    initialize_gene_pool();
}

// simulate one layout
static double simulate_grid(int *grid) {
    // reset buffers
    memset(progress_arr, 0, sim_size * sizeof(double));
    memset(time_req_arr, 0, sim_size * sizeof(int));
    memset(fertilized_arr, 0, sim_size * sizeof(char));
    memset(watered_arr, 0, sim_size * sizeof(char));
    double harvested = 0.0;
    for (int tick = 0; tick < eval_ticks; tick++) {
        // planter
        for (int r = 0; r < sim_rows; r++) {
            int r0 = r % rows;
            for (int c = 0; c < sim_cols; c++) {
                if (GET(r,c) == ITEM_PLANTER) {
                    int side = side_arr[0], half = half_arr[0];
                    for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++) {
                        for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                            int idx = i*sim_cols + j;
                            if (GET(i,j) == ITEM_NONE && time_req_arr[idx] == 0) {
                                time_req_arr[idx] = grow_time;
                            }
                        }
                    }
                }
            }
        }
        // fertilizer and sprinkler
        for (int r = 0; r < sim_rows; r++) {
            int r0 = r % rows;
            for (int c = 0; c < sim_cols; c++) {
                int t = GET(r,c);
                if (t == ITEM_FERTILIZER) {
                    int side = side_arr[1], half = half_arr[1];
                    for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++) {
                        for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                            int idx = i*sim_cols + j;
                            if (time_req_arr[idx] > 0 && !fertilized_arr[idx]) {
                                fertilized_arr[idx] = 1;
                                if (watered_arr[idx]) time_req_arr[idx] = (int)(grow_time * 0.5);
                                else time_req_arr[idx] = (int)(grow_time * 0.75);
                            }
                        }
                    }
                }
                if (t == ITEM_SPRINKLER) {
                    int side = side_arr[2], half = half_arr[2];
                    for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++) {
                        for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                            int idx = i*sim_cols + j;
                            if (time_req_arr[idx] > 0 && !watered_arr[idx]) {
                                watered_arr[idx] = 1;
                                if (fertilized_arr[idx]) time_req_arr[idx] = (int)(grow_time * 0.5);
                                else time_req_arr[idx] = (int)(grow_time * 0.75);
                            }
                        }
                    }
                }
            }
        }
        // growth
        for (int i = 0; i < sim_rows*sim_cols; i++) {
            if (time_req_arr[i] > 0) progress_arr[i] += 1.0 / time_req_arr[i];
        }
        // harvester
        for (int r = 0; r < sim_rows; r++) {
            int r0 = r % rows;
            for (int c = 0; c < sim_cols; c++) {
                if (GET(r,c) == ITEM_HARVESTER) {
                    int side = side_arr[3], half = half_arr[3];
                    for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++) {
                        for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                            int idx = i*sim_cols + j;
                            if (progress_arr[idx] >= 1.0) {
                                harvested += 1.0;
                                progress_arr[idx] = 0.0;
                                time_req_arr[idx] = 0;
                                fertilized_arr[idx] = 0;
                                watered_arr[idx] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    // placement cost
    double cost_penalty = 0.0;
    for (int i = 0; i < sim_rows*sim_cols; i++) {
        int t = GET(i%sim_rows, i%sim_cols);
        if (t > ITEM_NONE) cost_penalty += costs_arr[t-1];
    }
    // unreachable squares penalty
    int unreach_planter = 0, unreach_harv = 0;
    int side_p = side_arr[0], half_p = half_arr[0];
    int side_h = side_arr[3], half_h = half_arr[3];
    for (int r = 0; r < sim_rows; r++) {
        for (int c = 0; c < sim_cols; c++) {
            int reach = 0;
            for (int dr = -half_p; dr <= half_p && !reach; dr++) {
                for (int dc = -half_p; dc <= half_p; dc++) {
                    int pr = r + dr, pc = c + dc;
                    if (pr >= 0 && pr < sim_rows && pc >= 0 && pc < sim_cols && GET(pr,pc) == ITEM_PLANTER) {
                        reach = 1; break;
                    }
                }
            }
            if (!reach) unreach_planter++;
            reach = 0;
            for (int dr = -half_h; dr <= half_h && !reach; dr++) {
                for (int dc = -half_h; dc <= half_h; dc++) {
                    int pr = r + dr, pc = c + dc;
                    if (pr >= 0 && pr < sim_rows && pc >= 0 && pc < sim_cols && GET(pr,pc) == ITEM_HARVESTER) {
                        reach = 1; break;
                    }
                }
            }
            if (!reach) unreach_harv++;
        }
    }
    double net_score = harvested - cost_penalty;
    double penalty_factor = 1.0 - 0.05 * unreach_planter - 0.05 * unreach_harv;
    if (penalty_factor < 0) penalty_factor = 0;
    double final_score = net_score * penalty_factor;
    // no per-call frees; buffers reused
    return final_score;
}

// run GA
DLL_EXPORT void run_optimizer(int *out_best_grid, double *out_best_fitness) {
    int stagnant_count = 0;
    double prev_best = best_fitness_global;
    for (int gen = 0; gen < generations_count; gen++) {
        // evaluate fitness
        for (int i = 0; i < pop_size; i++) fitness_arr[i] = simulate_grid(population_arr[i]);
        // update transposition table
        for (int i = 0; i < pop_size; i++) {
            if (!is_known_gene(population_arr[i]))
                add_gene_entry(population_arr[i], fitness_arr[i]);
        }
        purge_gene_pool();
        // select parents via Gumbel-top-k soft selection
        int parent_count = pop_size / 2;
        int k1 = pop_size / 4;
        int k2 = parent_count - k1;
        int *sel = malloc(pop_size * sizeof(int));
        double *pert = malloc(pop_size * sizeof(double));
        for (int i = 0; i < pop_size; i++) sel[i] = i;
        double max_phi = fitness_arr[0];
        for (int i = 1; i < pop_size; i++) if (fitness_arr[i] > max_phi) max_phi = fitness_arr[i];
        // sample first k1 (T=1.0)
        for (int s = 0; s < k1; s++) {
            for (int j = s; j < pop_size; j++) {
                int idx = sel[j];
                double g = -log(-log(rand_double()));
                pert[j] = (fitness_arr[idx] - max_phi) + g;
            }
            int best_j = s;
            for (int j = s+1; j < pop_size; j++) if (pert[j] > pert[best_j]) best_j = j;
            int tmp = sel[s]; sel[s] = sel[best_j]; sel[best_j] = tmp;
        }
        // sample next k2 (T=1.25)
        double T = 1.25;
        for (int s = k1; s < parent_count; s++) {
            for (int j = s; j < pop_size; j++) {
                int idx = sel[j];
                double g = -log(-log(rand_double()));
                pert[j] = (fitness_arr[idx] - max_phi) / T + g;
            }
            int best_j2 = s;
            for (int j = s+1; j < pop_size; j++) if (pert[j] > pert[best_j2]) best_j2 = j;
            int tmp2 = sel[s]; sel[s] = sel[best_j2]; sel[best_j2] = tmp2;
        }
        int *parent_idxs = malloc(parent_count * sizeof(int));
        for (int i = 0; i < parent_count; i++) parent_idxs[i] = sel[i];
        free(pert);
        free(sel);
        // update best individual
        int best_i = 0;
        for (int i = 1; i < pop_size; i++) if (fitness_arr[i] > fitness_arr[best_i]) best_i = i;
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
        if (stagnant_count >= 50) {
            break;
        }
        // create new population
        int **newpop = malloc(pop_size * sizeof(int*));
        for (int i = 0; i < parent_count; i++) {
            newpop[i] = allocate_grid();
            copy_grid(newpop[i], population_arr[parent_idxs[i]]);
        }
        // breeding
        for (int i = parent_count; i < pop_size; i++) {
            // select parents
            double *weights = malloc(pop_size * sizeof(double));
            double total = 0.0;
            for (int j = 0; j < pop_size; j++) {
                double w = fitness_arr[j] > 0.0 ? fitness_arr[j] : 0.0;
                weights[j] = w; total += w;
            }
            int p1, p2;
            if (total <= 0.0) {
                p1 = rand_int(pop_size); p2 = rand_int(pop_size);
            } else {
                double r1 = rand_double() * total;
                double sum = 0.0;
                for (p1 = 0; p1 < pop_size; p1++) { sum += weights[p1]; if (sum >= r1) break; }
                double r2 = rand_double() * total;
                sum = 0.0;
                for (p2 = 0; p2 < pop_size; p2++) { sum += weights[p2]; if (sum >= r2) break; }
            }
            free(weights);
            // crossover
            int *child = allocate_grid();
            for (int k = 0; k < rows*cols; k++) {
                child[k] = (rand_double() < cross_rate) ? population_arr[p1][k] : population_arr[p2][k];
            }
            // mutate
            for (int k = 0; k < rows*cols; k++) {
                if (rand_double() < mutation_rate) child[k] = rand_int(5);
            }
            // if child seen before, force heavy mutation
            if (is_known_gene(child)) {
                for (int k = 0; k < rows*cols; k++) {
                    if (rand_double() < 0.5) child[k] = rand_int(5);
                }
            }
            newpop[i] = child;
        }
        // replace old population
        for (int i = 0; i < pop_size; i++) free(population_arr[i]);
        free(population_arr); population_arr = newpop;
        free(parent_idxs);
    }
    // output
    *out_best_fitness = best_fitness_global;
    copy_grid(out_best_grid, best_grid_global);
}

// cleanup
DLL_EXPORT void free_optimizer() {
    for (int i = 0; i < pop_size; i++) free(population_arr[i]);
    free(population_arr); free(fitness_arr); free(best_grid_global);
    for (int i = 0; i < table_size; i++) {
        HashNode *cur = gene_table[i];
        while (cur) {
            HashNode *next = cur->next;
            free(cur->grid);
            free(cur);
            cur = next;
        }
    }
    free(gene_table);
    // free simulation buffers
    free(progress_arr);
    free(time_req_arr);
    free(fertilized_arr);
    free(watered_arr);
}
