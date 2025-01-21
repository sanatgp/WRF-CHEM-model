#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

typedef double real_t;

typedef struct {
    int nx, ny, nz;
    int px, py;
    int iterations;
    int horz_order;
    int vert_order; 
    int polar;
    int periodic_x;
    int periodic_y;
    int symmetric_xs;
    int symmetric_xe;
    int symmetric_ys;
    int symmetric_ye;
} Config;

typedef struct {
    real_t *msftx, *msfty;
    real_t *msfux, *msfuy;
    real_t *msfvx, *msfvy;
    real_t *w, *w_old;  // Using w instead of u to match Fortran
    real_t *tendency;
    real_t *z_tendency;
    real_t *h_tendency;
    real_t *ru, *rv, *rom;
    real_t dx, dy, dz;
    real_t *fzm, *fzp;
    real_t *rdzw;
    real_t *c1, *c2;  // Additional coefficients from Fortran
    real_t *mut;      // Additional field from Fortran
    
    real_t *fqx, *fqy, *fqz;
    real_t *fqxl, *fqyl, *fqzl;
    
    int rank, size;
    int dims[2];
    int coords[2];
    MPI_Comm cart_comm;
    
    int local_nx, local_ny, local_nz;
    int ghost_width;
    real_t *send_east, *send_west;
    real_t *send_north, *send_south;
    real_t *recv_east, *recv_west;
    real_t *recv_north, *recv_south;
    
    Config config;
} Grid;

#define IDX3D(i,j,k) (((k)*grid->local_ny + (j))*grid->local_nx + (i))

// Flux operators make sure matching WRF exactly
real_t flux2(real_t q_im1, real_t q_i, real_t vel) {
    return 0.5 * vel * (q_i + q_im1);
}

real_t flux4(real_t q_im2, real_t q_im1, real_t q_i, real_t q_ip1, real_t vel) {
    return vel * ((7.0/12.0)*(q_i + q_im1) - (1.0/12.0)*(q_ip1 + q_im2));
}

real_t flux3(real_t q_im2, real_t q_im1, real_t q_i, real_t q_ip1, real_t vel, int time_step) {
    return flux4(q_im2, q_im1, q_i, q_ip1, vel) + 
           copysign(1.0, (real_t)time_step) * copysign(1.0, vel) * 
           ((q_ip1 - q_im2) - 3.0*(q_i - q_im1)) / 12.0;
}

real_t flux6(real_t q_im3, real_t q_im2, real_t q_im1, real_t q_i, 
             real_t q_ip1, real_t q_ip2, real_t vel) {
    return vel * ((37.0/60.0)*(q_i + q_im1) - (8.0/60.0)*(q_ip1 + q_im2) 
                  + (1.0/60.0)*(q_ip2 + q_im3));
}

real_t flux5(real_t q_im3, real_t q_im2, real_t q_im1, real_t q_i,
             real_t q_ip1, real_t q_ip2, real_t vel, int time_step) {
    return flux6(q_im3, q_im2, q_im1, q_i, q_ip1, q_ip2, vel) -
           copysign(1.0, (real_t)time_step) * copysign(1.0, vel) *
           ((q_ip2 - q_im3) - 5.0*(q_ip1 - q_im2) + 10.0*(q_i - q_im1)) / 60.0;
}

void calculate_horizontal_advection(Grid *grid, int time_step) {
    const int horz_order = grid->config.horz_order;
    
    // Y-direction first (matching WRF)
    for(int j = grid->ghost_width; j < grid->local_ny + grid->ghost_width; j++) {
        for(int k = 1; k < grid->local_nz; k++) {
            for(int i = grid->ghost_width; i < grid->local_nx + grid->ghost_width; i++) {
                real_t vel = grid->rv[IDX3D(i,k,j)];
                real_t flux;
                
                if(horz_order == 6) {
                    flux = flux6(grid->w[IDX3D(i,k,j-3)],
                                grid->w[IDX3D(i,k,j-2)],
                                grid->w[IDX3D(i,k,j-1)],
                                grid->w[IDX3D(i,k,j)],
                                grid->w[IDX3D(i,k,j+1)],
                                grid->w[IDX3D(i,k,j+2)],
                                vel);
                }
                else if(horz_order == 5) {
                    flux = flux5(grid->w[IDX3D(i,k,j-3)],
                                grid->w[IDX3D(i,k,j-2)],
                                grid->w[IDX3D(i,k,j-1)],
                                grid->w[IDX3D(i,k,j)],
                                grid->w[IDX3D(i,k,j+1)],
                                grid->w[IDX3D(i,k,j+2)],
                                vel, time_step);
                }
                else if(horz_order == 4) {
                    flux = flux4(grid->w[IDX3D(i,k,j-2)],
                                grid->w[IDX3D(i,k,j-1)],
                                grid->w[IDX3D(i,k,j)],
                                grid->w[IDX3D(i,k,j+1)],
                                vel);
                }
                else if(horz_order == 3) {
                    flux = flux3(grid->w[IDX3D(i,k,j-2)],
                                grid->w[IDX3D(i,k,j-1)],
                                grid->w[IDX3D(i,k,j)],
                                grid->w[IDX3D(i,k,j+1)],
                                vel, time_step);
                }
                else {
                    flux = flux2(grid->w[IDX3D(i,k,j-1)],
                                grid->w[IDX3D(i,k,j)],
                                vel);
                }
                
                grid->fqy[IDX3D(i,k,j)] = flux;
            }
        }
    }

    // X-direction second (matching WRF)
    for(int j = grid->ghost_width; j < grid->local_ny + grid->ghost_width; j++) {
        for(int k = 1; k < grid->local_nz; k++) {
            for(int i = grid->ghost_width; i < grid->local_nx + grid->ghost_width; i++) {
                real_t vel = grid->ru[IDX3D(i,k,j)];
                real_t flux;
                
                if(horz_order == 6) {
                    flux = flux6(grid->w[IDX3D(i-3,k,j)],
                                grid->w[IDX3D(i-2,k,j)],
                                grid->w[IDX3D(i-1,k,j)],
                                grid->w[IDX3D(i,k,j)],
                                grid->w[IDX3D(i+1,k,j)],
                                grid->w[IDX3D(i+2,k,j)],
                                vel);
                }
                else if(horz_order == 5) {
                    flux = flux5(grid->w[IDX3D(i-3,k,j)],
                                grid->w[IDX3D(i-2,k,j)],
                                grid->w[IDX3D(i-1,k,j)],
                                grid->w[IDX3D(i,k,j)],
                                grid->w[IDX3D(i+1,k,j)],
                                grid->w[IDX3D(i+2,k,j)],
                                vel, time_step);
                }
                else if(horz_order == 4) {
                    flux = flux4(grid->w[IDX3D(i-2,k,j)],
                                grid->w[IDX3D(i-1,k,j)],
                                grid->w[IDX3D(i,k,j)],
                                grid->w[IDX3D(i+1,k,j)],
                                vel);
                }
                else if(horz_order == 3) {
                    flux = flux3(grid->w[IDX3D(i-2,k,j)],
                                grid->w[IDX3D(i-1,k,j)],
                                grid->w[IDX3D(i,k,j)],
                                grid->w[IDX3D(i+1,k,j)],
                                vel, time_step);
                }
                else {
                    flux = flux2(grid->w[IDX3D(i-1,k,j)],
                                grid->w[IDX3D(i,k,j)],
                                vel);
                }
                
                grid->fqx[IDX3D(i,k,j)] = flux;
            }
        }
    }
}

void calculate_vertical_advection(Grid *grid, int time_step) {
    const int vert_order = grid->config.vert_order;
    
    for(int j = grid->ghost_width; j < grid->local_ny + grid->ghost_width; j++) {
        for(int i = grid->ghost_width; i < grid->local_nx + grid->ghost_width; i++) {
            // Zero flux at boundaries
            grid->fqz[IDX3D(i,0,j)] = 0.0;
            grid->fqz[IDX3D(i,grid->local_nz,j)] = 0.0;
            
            for(int k = 1; k < grid->local_nz; k++) {
                real_t vel = grid->rom[IDX3D(i,k,j)];
                real_t flux;
                
                if(vert_order == 6) {
                    if(k >= 3 && k <= grid->local_nz-3) {
                        flux = flux6(grid->w[IDX3D(i,k-3,j)],
                                    grid->w[IDX3D(i,k-2,j)],
                                    grid->w[IDX3D(i,k-1,j)],
                                    grid->w[IDX3D(i,k,j)],
                                    grid->w[IDX3D(i,k+1,j)],
                                    grid->w[IDX3D(i,k+2,j)],
                                    -vel);
                    } else {
                        flux = flux2(grid->w[IDX3D(i,k-1,j)],
                                    grid->w[IDX3D(i,k,j)],
                                    -vel);
                    }
                }
                else if(vert_order == 5) {
                    if(k >= 3 && k <= grid->local_nz-3) {
                        flux = flux5(grid->w[IDX3D(i,k-3,j)],
                                    grid->w[IDX3D(i,k-2,j)],
                                    grid->w[IDX3D(i,k-1,j)],
                                    grid->w[IDX3D(i,k,j)],
                                    grid->w[IDX3D(i,k+1,j)],
                                    grid->w[IDX3D(i,k+2,j)],
                                    -vel, time_step);
                    } else {
                        flux = flux2(grid->w[IDX3D(i,k-1,j)],
                                    grid->w[IDX3D(i,k,j)],
                                    -vel);
                    }
                }
                else if(vert_order == 4) {
                    if(k >= 2 && k <= grid->local_nz-2) {
                        flux = flux4(grid->w[IDX3D(i,k-2,j)],
                                    grid->w[IDX3D(i,k-1,j)],
                                    grid->w[IDX3D(i,k,j)],
                                    grid->w[IDX3D(i,k+1,j)],
                                    -vel);
                    } else {
                        flux = flux2(grid->w[IDX3D(i,k-1,j)],
                                    grid->w[IDX3D(i,k,j)],
                                    -vel);
                    }
                }
                else if(vert_order == 3) {
                    if(k >= 2 && k <= grid->local_nz-2) {
                        flux = flux3(grid->w[IDX3D(i,k-2,j)],
                                    grid->w[IDX3D(i,k-1,j)],
                                    grid->w[IDX3D(i,k,j)],
                                    grid->w[IDX3D(i,k+1,j)],
                                    -vel, time_step);
                    } else {
                        flux = flux2(grid->w[IDX3D(i,k-1,j)],
                                    grid->w[IDX3D(i,k,j)],
                                    -vel);
                    }
                }
                else {  // 2nd order
                    flux = flux2(grid->w[IDX3D(i,k-1,j)],
                                grid->w[IDX3D(i,k,j)],
                                -vel);
                }
                
                grid->fqz[IDX3D(i,k,j)] = flux;
            }
        }
    }
}

void apply_tendency(Grid *grid) {
    const real_t rdx = 1.0 / grid->dx;
    const real_t rdy = 1.0 / grid->dy;
    
    for(int j = grid->ghost_width; j < grid->local_ny + grid->ghost_width; j++) {
        for(int k = 1; k < grid->local_nz; k++) {
            for(int i = grid->ghost_width; i < grid->local_nx + grid->ghost_width; i++) {
                // X flux contribution
                real_t mrdx = grid->msftx[j * grid->local_nx + i] * rdx;
                grid->tendency[IDX3D(i,k,j)] = 
                    -mrdx * (grid->fqx[IDX3D(i+1,k,j)] - grid->fqx[IDX3D(i,k,j)]);
                
                // Y flux contribution
                real_t mrdy = grid->msftx[j * grid->local_nx + i] * rdy;
                grid->tendency[IDX3D(i,k,j)] -= 
                    mrdy * (grid->fqy[IDX3D(i,k,j+1)] - grid->fqy[IDX3D(i,k,j)]);
                
                // Z flux contribution
                grid->tendency[IDX3D(i,k,j)] -= 
                    grid->rdzw[k] * (grid->fqz[IDX3D(i,k+1,j)] - grid->fqz[IDX3D(i,k,j)]);
            }
        }
    }
}


int init_grid(Grid *grid) {
    // Compute ghost width based on maximum stencil order
    grid->ghost_width = ((grid->config.horz_order > grid->config.vert_order ? 
                         grid->config.horz_order : grid->config.vert_order) + 1) / 2;
    
    int nx = grid->local_nx + 2 * grid->ghost_width;
    int ny = grid->local_ny + 2 * grid->ghost_width;
    int nz = grid->local_nz;
    
    // Allocate arrays
    size_t size = nx * ny * nz * sizeof(real_t);
    grid->w = (real_t*)malloc(size);
    grid->w_old = (real_t*)malloc(size);
    grid->tendency = (real_t*)malloc(size);
    grid->z_tendency = (real_t*)malloc(size);
    grid->h_tendency = (real_t*)malloc(size);
    grid->ru = (real_t*)malloc(size);
    grid->rv = (real_t*)malloc(size);
    grid->rom = (real_t*)malloc(size);
    
    grid->fqx = (real_t*)malloc(size);
    grid->fqy = (real_t*)malloc(size);
    grid->fqz = (real_t*)malloc(size);
    grid->fqxl = (real_t*)malloc(size);
    grid->fqyl = (real_t*)malloc(size);
    grid->fqzl = (real_t*)malloc(size);
    
    grid->fzm = (real_t*)malloc(nz * sizeof(real_t));
    grid->fzp = (real_t*)malloc(nz * sizeof(real_t));
    grid->rdzw = (real_t*)malloc(nz * sizeof(real_t));
    grid->c1 = (real_t*)malloc(nz * sizeof(real_t));
    grid->c2 = (real_t*)malloc(nz * sizeof(real_t));
    
    size_t map_size = nx * ny * sizeof(real_t);
    grid->msftx = (real_t*)malloc(map_size);
    grid->msfty = (real_t*)malloc(map_size);
    grid->msfux = (real_t*)malloc(map_size);
    grid->msfuy = (real_t*)malloc(map_size);
    grid->msfvx = (real_t*)malloc(map_size);
    grid->msfvy = (real_t*)malloc(map_size);
    grid->mut = (real_t*)malloc(map_size);
    
    // Allocate halo exchange buffers
    size_t halo_size_x = grid->ghost_width * grid->local_ny * grid->local_nz * sizeof(real_t);
    size_t halo_size_y = grid->ghost_width * grid->local_nx * grid->local_nz * sizeof(real_t);
    
    grid->send_east = (real_t*)malloc(halo_size_x);
    grid->send_west = (real_t*)malloc(halo_size_x);
    grid->recv_east = (real_t*)malloc(halo_size_x);
    grid->recv_west = (real_t*)malloc(halo_size_x);
    grid->send_north = (real_t*)malloc(halo_size_y);
    grid->send_south = (real_t*)malloc(halo_size_y);
    grid->recv_north = (real_t*)malloc(halo_size_y);
    grid->recv_south = (real_t*)malloc(halo_size_y);

    for(int k = 0; k < nz; k++) {
        for(int j = 0; j < ny; j++) {
            for(int i = 0; i < nx; i++) {
                int idx = IDX3D(i,j,k);
                grid->w[idx] = 1.0;
                grid->w_old[idx] = 1.0;
                grid->tendency[idx] = 0.0;
                grid->z_tendency[idx] = 0.0;
                grid->h_tendency[idx] = 0.0;
                grid->ru[idx] = 1.0;
                grid->rv[idx] = 1.0;
                grid->rom[idx] = 1.0;
                grid->fqx[idx] = 0.0;
                grid->fqy[idx] = 0.0;
                grid->fqz[idx] = 0.0;
                grid->fqxl[idx] = 0.0;
                grid->fqyl[idx] = 0.0;
                grid->fqzl[idx] = 0.0;
            }
        }
        grid->fzm[k] = 0.5;
        grid->fzp[k] = 0.5;
        grid->rdzw[k] = 1.0 / grid->dz;
        grid->c1[k] = 1.0;
        grid->c2[k] = 0.0;
    }

    // Initialize map factors and other 2D fields
    for(int j = 0; j < ny; j++) {
        for(int i = 0; i < nx; i++) {
            int idx = j * nx + i;
            grid->msftx[idx] = 1.0;
            grid->msfty[idx] = 1.0;
            grid->msfux[idx] = 1.0;
            grid->msfuy[idx] = 1.0;
            grid->msfvx[idx] = 1.0;
            grid->msfvy[idx] = 1.0;
            grid->mut[idx] = 1.0;
        }
    }

    grid->dx = 1.0;
    grid->dy = 1.0;
    grid->dz = 1.0;

    return 0;
}

void free_grid(Grid *grid) {
    free(grid->w);
    free(grid->w_old);
    free(grid->tendency);
    free(grid->z_tendency);
    free(grid->h_tendency);
    free(grid->ru);
    free(grid->rv);
    free(grid->rom);
    
    free(grid->fqx);
    free(grid->fqy);
    free(grid->fqz);
    free(grid->fqxl);
    free(grid->fqyl);
    free(grid->fqzl);
    
    free(grid->fzm);
    free(grid->fzp);
    free(grid->rdzw);
    free(grid->c1);
    free(grid->c2);
    
    free(grid->msftx);
    free(grid->msfty);
    free(grid->msfux);
    free(grid->msfuy);
    free(grid->msfvx);
    free(grid->msfvy);
    free(grid->mut);
    
    free(grid->send_east);
    free(grid->send_west);
    free(grid->send_north);
    free(grid->send_south);
    free(grid->recv_east);
    free(grid->recv_west);
    free(grid->recv_north);
    free(grid->recv_south);
    
    MPI_Comm_free(&grid->cart_comm);
}

void exchange_halos(Grid *grid) {
    MPI_Status status;
    int rank_east, rank_west, rank_north, rank_south;
    
    MPI_Cart_shift(grid->cart_comm, 0, 1, &rank_west, &rank_east);
    MPI_Cart_shift(grid->cart_comm, 1, 1, &rank_south, &rank_north);
    
    // Pack east/west halos
    for(int k = 0; k < grid->local_nz; k++) {
        for(int j = 0; j < grid->local_ny; j++) {
            for(int g = 0; g < grid->ghost_width; g++) {
                int src_idx = IDX3D(grid->local_nx-grid->ghost_width+g, j+grid->ghost_width, k);
                int dst_idx = k*grid->local_ny*grid->ghost_width + j*grid->ghost_width + g;
                grid->send_east[dst_idx] = grid->w[src_idx];
                
                src_idx = IDX3D(grid->ghost_width+g, j+grid->ghost_width, k);
                grid->send_west[dst_idx] = grid->w[src_idx];
            }
        }
    }

    // Exchange east/west
    if(rank_east >= 0 || grid->config.periodic_x) {
        MPI_Sendrecv(grid->send_east, grid->ghost_width*grid->local_ny*grid->local_nz, MPI_DOUBLE,
                     rank_east, 0, grid->recv_west, grid->ghost_width*grid->local_ny*grid->local_nz,
                     MPI_DOUBLE, rank_west, 0, grid->cart_comm, &status);
    }
    
    if(rank_west >= 0 || grid->config.periodic_x) {
        MPI_Sendrecv(grid->send_west, grid->ghost_width*grid->local_ny*grid->local_nz, MPI_DOUBLE,
                     rank_west, 0, grid->recv_east, grid->ghost_width*grid->local_ny*grid->local_nz,
                     MPI_DOUBLE, rank_east, 0, grid->cart_comm, &status);
    }
    
    // Pack north/south halos
    for(int k = 0; k < grid->local_nz; k++) {
        for(int i = 0; i < grid->local_nx; i++) {
            for(int g = 0; g < grid->ghost_width; g++) {
                int src_idx = IDX3D(i+grid->ghost_width, grid->local_ny-grid->ghost_width+g, k);
                int dst_idx = k*grid->local_nx*grid->ghost_width + i*grid->ghost_width + g;
                grid->send_north[dst_idx] = grid->w[src_idx];
                
                src_idx = IDX3D(i+grid->ghost_width, grid->ghost_width+g, k);
                grid->send_south[dst_idx] = grid->w[src_idx];
            }
        }
    }
    
    // Exchange north/south
    if(rank_north >= 0 || grid->config.periodic_y) {
        MPI_Sendrecv(grid->send_north, grid->ghost_width*grid->local_nx*grid->local_nz, MPI_DOUBLE,
                     rank_north, 0, grid->recv_south, grid->ghost_width*grid->local_nx*grid->local_nz,
                     MPI_DOUBLE, rank_south, 0, grid->cart_comm, &status);
    }
    
    if(rank_south >= 0 || grid->config.periodic_y) {
        MPI_Sendrecv(grid->send_south, grid->ghost_width*grid->local_nx*grid->local_nz, MPI_DOUBLE,
                     rank_south, 0, grid->recv_north, grid->ghost_width*grid->local_nx*grid->local_nz,
                     MPI_DOUBLE, rank_north, 0, grid->cart_comm, &status);
    }

    MPI_Barrier(grid->cart_comm);
    
    if(rank_east >= 0 || grid->config.periodic_x) {
        for(int k = 0; k < grid->local_nz; k++) {
            for(int j = 0; j < grid->local_ny; j++) {
                for(int g = 0; g < grid->ghost_width; g++) {
                    int src_idx = k*grid->local_ny*grid->ghost_width + j*grid->ghost_width + g;
                    int dst_idx = IDX3D(grid->local_nx+g, j+grid->ghost_width, k);
                    grid->w[dst_idx] = grid->recv_east[src_idx];
                }
            }
        }
    }
    
    if(rank_west >= 0 || grid->config.periodic_x) {
        for(int k = 0; k < grid->local_nz; k++) {
            for(int j = 0; j < grid->local_ny; j++) {
                for(int g = 0; g < grid->ghost_width; g++) {
                    int src_idx = k*grid->local_ny*grid->ghost_width + j*grid->ghost_width + g;
                    int dst_idx = IDX3D(g, j+grid->ghost_width, k);
                    grid->w[dst_idx] = grid->recv_west[src_idx];
                }
            }
        }
    }
    
    if(rank_north >= 0 || grid->config.periodic_y) {
        for(int k = 0; k < grid->local_nz; k++) {
            for(int i = 0; i < grid->local_nx; i++) {
                for(int g = 0; g < grid->ghost_width; g++) {
                    int src_idx = k*grid->local_nx*grid->ghost_width + i*grid->ghost_width + g;
                    int dst_idx = IDX3D(i+grid->ghost_width, grid->local_ny+g, k);
                    grid->w[dst_idx] = grid->recv_north[src_idx];
                }
            }
        }
    }
    
    if(rank_south >= 0 || grid->config.periodic_y) {
        for(int k = 0; k < grid->local_nz; k++) {
            for(int i = 0; i < grid->local_nx; i++) {
                for(int g = 0; g < grid->ghost_width; g++) {
                    int src_idx = k*grid->local_nx*grid->ghost_width + i*grid->ghost_width + g;
                    int dst_idx = IDX3D(i+grid->ghost_width, g, k);
                    grid->w[dst_idx] = grid->recv_south[src_idx];
                }
            }
        }
    }
}


void apply_stencils(Grid *grid, int time_step, double *total_time_ms) {
   double start_total, end_total;
   
   start_total = MPI_Wtime();
   
   exchange_halos(grid);
   calculate_horizontal_advection(grid, time_step);  
   calculate_vertical_advection(grid, time_step);
   apply_tendency(grid);
   
   end_total = MPI_Wtime();
   *total_time_ms = (end_total - start_total) * 1000.0;
}

void benchmark_stencils(Grid *grid) {
   double local_total_time_ms = 0.0;
   double iter_total_ms = 0.0;
   double max_total_time_ms;
   double total_comm_ms = 0.0, total_comp_ms = 0.0;
   double max_comm_ms = 0.0, max_comp_ms = 0.0;
   double start_phase, end_phase;
   
   if(grid->rank == 0) {
       printf("\nRunning advection stencil benchmark:\n");
       printf("Grid size: %d x %d x %d\n", grid->config.nx, grid->config.ny, grid->config.nz);
       printf("Process grid: %d x %d\n", grid->dims[0], grid->dims[1]);
       printf("Iterations: %d\n", grid->config.iterations);
       printf("Horizontal stencil order: %d\n", grid->config.horz_order);
       printf("Vertical stencil order: %d\n", grid->config.vert_order);
   }
   
   for(int i = 0; i < 3; i++) {
       apply_stencils(grid, i, &iter_total_ms);
   }
   
   MPI_Barrier(grid->cart_comm);
   
   start_phase = MPI_Wtime();
   for(int i = 0; i < grid->config.iterations; i++) {
       exchange_halos(grid);
   }
   end_phase = MPI_Wtime();
   total_comm_ms = (end_phase - start_phase) * 1000.0;
   
   start_phase = MPI_Wtime();
   for(int i = 0; i < grid->config.iterations; i++) {
       calculate_horizontal_advection(grid, i);
       calculate_vertical_advection(grid, i);
       apply_tendency(grid);
   }
   end_phase = MPI_Wtime();
   total_comp_ms = (end_phase - start_phase) * 1000.0;
   
   for(int iter = 0; iter < grid->config.iterations; iter++) {
       apply_stencils(grid, iter, &iter_total_ms);
       local_total_time_ms += iter_total_ms;
   }
   
   MPI_Reduce(&local_total_time_ms, &max_total_time_ms, 1, MPI_DOUBLE, MPI_MAX, 0, grid->cart_comm);
   MPI_Reduce(&total_comm_ms, &max_comm_ms, 1, MPI_DOUBLE, MPI_MAX, 0, grid->cart_comm);
   MPI_Reduce(&total_comp_ms, &max_comp_ms, 1, MPI_DOUBLE, MPI_MAX, 0, grid->cart_comm);
   
   if(grid->rank == 0) {
       double avg_total_ms = max_total_time_ms / grid->config.iterations;
       double avg_comm_ms = max_comm_ms / grid->config.iterations;
       double avg_comp_ms = max_comp_ms / grid->config.iterations;
       double avg_other_ms = avg_total_ms - (avg_comm_ms + avg_comp_ms);
       
       double total_cells = (double)grid->config.nx * grid->config.ny * grid->config.nz;
       double cells_per_ms = total_cells / avg_total_ms;
       
       int flops_per_point = 
           (grid->config.horz_order * 10) * 2 + 
           (grid->config.vert_order * 10);
       double gflops = (cells_per_ms * flops_per_point) / 1e6;
       
       printf("\nTiming Results (per iteration):\n");
       printf("Total time:     %9.3f ms\n", avg_total_ms);
       printf("Communication:  %9.3f ms (%.1f%%)\n", 
              avg_comm_ms, (avg_comm_ms/avg_total_ms)*100);
       printf("Computation:    %9.3f ms (%.1f%%)\n", 
              avg_comp_ms, (avg_comp_ms/avg_total_ms)*100);
       printf("Other:          %9.3f ms (%.1f%%)\n",
              avg_other_ms, (avg_other_ms/avg_total_ms)*100);
       
       printf("\nPerformance Metrics:\n");
       printf("Cells/ms:       %.2e\n", cells_per_ms);
       printf("Est. GFLOPS:    %.2f\n", gflops);
       printf("Memory usage:   %.2f GB per rank\n",
              (double)(grid->local_nx * grid->local_ny * grid->local_nz * 15 * sizeof(real_t)) / 
              (1024*1024*1024));
   }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    Grid grid;
    memset(&grid, 0, sizeof(Grid));
    

    if(argc != 9) {
        if(grid.rank == 0) {
            printf("Usage: %s nx ny nz px py iters horz_order vert_order\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    grid.config.nx = atoi(argv[1]);
    grid.config.ny = atoi(argv[2]);
    grid.config.nz = atoi(argv[3]);
    grid.config.px = atoi(argv[4]);
    grid.config.py = atoi(argv[5]);
    grid.config.iterations = atoi(argv[6]);
    grid.config.horz_order = atoi(argv[7]);
    grid.config.vert_order = atoi(argv[8]);
    
    // Initialize MPI topology
    MPI_Comm_rank(MPI_COMM_WORLD, &grid.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &grid.size);
    
    if(grid.size != grid.config.px * grid.config.py) {
        if(grid.rank == 0) {
            printf("Error: Number of processes (%d) must match px*py (%d)\n",
                   grid.size, grid.config.px * grid.config.py);
        }
        MPI_Finalize();
        return 1;
    }
    
    grid.dims[0] = grid.config.px;
    grid.dims[1] = grid.config.py;
    int periods[2] = {grid.config.periodic_x, grid.config.periodic_y};
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid.dims, periods, 0, &grid.cart_comm);
    MPI_Cart_coords(grid.cart_comm, grid.rank, 2, grid.coords);
    
    grid.local_nx = grid.config.nx / grid.config.px;
    grid.local_ny = grid.config.ny / grid.config.py;
    grid.local_nz = grid.config.nz;
    
    if(grid.local_nx * grid.config.px != grid.config.nx ||
       grid.local_ny * grid.config.py != grid.config.ny) {
        if(grid.rank == 0) {
            printf("Error: Grid dimensions must be divisible by process grid\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if(init_grid(&grid) != 0) {
        MPI_Finalize();
        return 1;
    }
    
    benchmark_stencils(&grid);
    
    free_grid(&grid);
    MPI_Finalize();
    return 0;
}