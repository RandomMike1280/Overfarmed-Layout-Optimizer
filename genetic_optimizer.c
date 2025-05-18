#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

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

// forward declarations for utilities used in gene pool
static int *allocate_grid();
static void copy_grid(int *dest, int *src);

// gene pool support
typedef struct {
    int *grid;
    double fitness;
} GeneEntry;
static GeneEntry *gene_pool = NULL;
static int gene_pool_size = 0;
static int gene_pool_capacity = 0;

static void initialize_gene_pool() {
    gene_pool_capacity = pop_size * 10;
    gene_pool = malloc(gene_pool_capacity * sizeof(GeneEntry));
    gene_pool_size = 0;
}

static int is_known_gene(int *grid) {
    for (int i = 0; i < gene_pool_size; i++) {
        if (memcmp(gene_pool[i].grid, grid, rows*cols*sizeof(int)) == 0) return 1;
    }
    return 0;
}

static void add_gene_entry(int *grid, double fitness) {
    if (!gene_pool) initialize_gene_pool();
    if (gene_pool_size >= gene_pool_capacity) {
        gene_pool_capacity *= 2;
        gene_pool = realloc(gene_pool, gene_pool_capacity * sizeof(GeneEntry));
    }
    int *gcopy = allocate_grid();
    copy_grid(gcopy, grid);
    gene_pool[gene_pool_size].grid = gcopy;
    gene_pool[gene_pool_size].fitness = fitness;
    gene_pool_size++;
}

static void purge_gene_pool() {
    double threshold = best_fitness_global * 0.5;
    for (int i = 0; i < gene_pool_size; i++) {
        if (gene_pool[i].fitness < threshold) {
            free(gene_pool[i].grid);
            gene_pool[i] = gene_pool[--gene_pool_size];
            i--;
        }
    }
}

// utilities
static double rand_double() { return rand() / (double)RAND_MAX; }
static int rand_int(int max) { return rand() % max; }
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
    for (int i = 0; i < 4; i++) {
        ranges_arr[i] = _ranges[i]; costs_arr[i] = _costs[i];
    }
    // seed RNG
    srand((unsigned)time(NULL));
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
    int sim_rows = rows * 3;
    int sim_cols = cols * 3;
    // tile base grid into 3x3 layout
    int *sim_grid = malloc(sim_rows * sim_cols * sizeof(int));
    for (int ti = 0; ti < 3; ti++) {
        for (int tj = 0; tj < 3; tj++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    sim_grid[(ti*rows + r) * sim_cols + (tj*cols + c)] = grid[r*cols + c];
                }
            }
        }
    }
    double *progress = calloc(sim_rows * sim_cols, sizeof(double));
    int *time_req = calloc(sim_rows * sim_cols, sizeof(int));
    char *fertilized = calloc(sim_rows * sim_cols, 1);
    char *watered = calloc(sim_rows * sim_cols, 1);
    double harvested = 0.0;
    for (int tick = 0; tick < eval_ticks; tick++) {
        // planter
        for (int r = 0; r < sim_rows; r++) {
            for (int c = 0; c < sim_cols; c++) {
                if (sim_grid[r*sim_cols + c] == ITEM_PLANTER) {
                    int side = (int)ranges_arr[0], half = side/2;
                    for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++)
                        for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                            int idx = i*sim_cols + j;
                            if (sim_grid[idx] == ITEM_NONE && time_req[idx] == 0) {
                                time_req[idx] = grow_time;
                            }
                        }
                }
            }
        }
        // fertilizer and sprinkler
        for (int r = 0; r < sim_rows; r++) {
            for (int c = 0; c < sim_cols; c++) {
                int t = sim_grid[r*sim_cols + c];
                if (t == ITEM_FERTILIZER) {
                    int side = (int)ranges_arr[1], half = side/2;
                    for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++)
                        for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                            int idx = i*sim_cols + j;
                            if (time_req[idx] > 0 && !fertilized[idx]) {
                                fertilized[idx] = 1;
                                if (watered[idx]) time_req[idx] = (int)(grow_time * 0.5);
                                else time_req[idx] = (int)(grow_time * 0.75);
                            }
                        }
                }
                if (t == ITEM_SPRINKLER) {
                    int side = (int)ranges_arr[2], half = side/2;
                    for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++)
                        for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                            int idx = i*sim_cols + j;
                            if (time_req[idx] > 0 && !watered[idx]) {
                                watered[idx] = 1;
                                if (fertilized[idx]) time_req[idx] = (int)(grow_time * 0.5);
                                else time_req[idx] = (int)(grow_time * 0.75);
                            }
                        }
                }
            }
        }
        // growth
        for (int i = 0; i < sim_rows*sim_cols; i++) {
            if (time_req[i] > 0) progress[i] += 1.0 / time_req[i];
        }
        // harvester
        for (int r = 0; r < sim_rows; r++) {
            for (int c = 0; c < sim_cols; c++) {
                if (sim_grid[r*sim_cols + c] == ITEM_HARVESTER) {
                    int side = (int)ranges_arr[3], half = side/2;
                    for (int i = MAX(0, r-half); i < MIN(sim_rows, r-half+side); i++)
                        for (int j = MAX(0, c-half); j < MIN(sim_cols, c-half+side); j++) {
                            int idx = i*sim_cols + j;
                            if (progress[idx] >= 1.0) {
                                harvested += 1.0;
                                progress[idx] = 0.0;
                                time_req[idx] = 0;
                                fertilized[idx] = 0;
                                watered[idx] = 0;
                            }
                        }
                }
            }
        }
    }
    // placement cost
    double cost_penalty = 0.0;
    for (int i = 0; i < sim_rows*sim_cols; i++) {
        int t = sim_grid[i];
        if (t > ITEM_NONE) cost_penalty += costs_arr[t-1];
    }
    // unreachable squares penalty
    int unreach_planter = 0, unreach_harv = 0;
    int side_p = (int)ranges_arr[0], half_p = side_p/2;
    int side_h = (int)ranges_arr[3], half_h = side_h/2;
    for (int r = 0; r < sim_rows; r++) {
        for (int c = 0; c < sim_cols; c++) {
            int reach = 0;
            for (int dr = -half_p; dr <= half_p && !reach; dr++) {
                for (int dc = -half_p; dc <= half_p; dc++) {
                    int pr = r + dr, pc = c + dc;
                    if (pr >= 0 && pr < sim_rows && pc >= 0 && pc < sim_cols && sim_grid[pr*sim_cols + pc] == ITEM_PLANTER) {
                        reach = 1; break;
                    }
                }
            }
            if (!reach) unreach_planter++;
            reach = 0;
            for (int dr = -half_h; dr <= half_h && !reach; dr++) {
                for (int dc = -half_h; dc <= half_h; dc++) {
                    int pr = r + dr, pc = c + dc;
                    if (pr >= 0 && pr < sim_rows && pc >= 0 && pc < sim_cols && sim_grid[pr*sim_cols + pc] == ITEM_HARVESTER) {
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
    free(progress);
    free(time_req);
    free(fertilized);
    free(watered);
    free(sim_grid);
    return final_score;
}

// run GA
DLL_EXPORT void run_optimizer(int *out_best_grid, double *out_best_fitness) {
    for (int gen = 0; gen < generations_count; gen++) {
        // evaluate fitness
        for (int i = 0; i < pop_size; i++) fitness_arr[i] = simulate_grid(population_arr[i]);
        // update transposition table
        for (int i = 0; i < pop_size; i++) {
            if (!is_known_gene(population_arr[i]))
                add_gene_entry(population_arr[i], fitness_arr[i]);
        }
        purge_gene_pool();
        // select elites
        int *idxs = malloc(pop_size * sizeof(int));
        for (int i = 0; i < pop_size; i++) idxs[i] = i;
        for (int i = 0; i < top_k; i++) {
            int max_i = i;
            for (int j = i+1; j < pop_size; j++) {
                if (fitness_arr[idxs[j]] > fitness_arr[idxs[max_i]]) max_i = j;
            }
            int tmp = idxs[i]; idxs[i] = idxs[max_i]; idxs[max_i] = tmp;
        }
        // update best
        if (fitness_arr[idxs[0]] > best_fitness_global) {
            best_fitness_global = fitness_arr[idxs[0]];
            copy_grid(best_grid_global, population_arr[idxs[0]]);
        }
        // create new population
        int **newpop = malloc(pop_size * sizeof(int*));
        for (int i = 0; i < top_k; i++) {
            newpop[i] = allocate_grid();
            copy_grid(newpop[i], population_arr[idxs[i]]);
        }
        // breeding
        for (int i = top_k; i < pop_size; i++) {
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
        free(population_arr);
        population_arr = newpop;
        free(idxs);
    }
    // output
    *out_best_fitness = best_fitness_global;
    copy_grid(out_best_grid, best_grid_global);
}

// cleanup
DLL_EXPORT void free_optimizer() {
    for (int i = 0; i < pop_size; i++) free(population_arr[i]);
    free(population_arr); free(fitness_arr); free(best_grid_global);
    for (int i = 0; i < gene_pool_size; i++) free(gene_pool[i].grid);
    free(gene_pool);
}
