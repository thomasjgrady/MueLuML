// Trilinos header files

// Belos
#include <BelosBlockCGSolMgr.hpp>
#include <BelosIteration.hpp>
#include <BelosMultiVec.hpp>
#include <BelosMultiVecTraits.hpp>
#include <BelosOperatorTraits.hpp>
#include <Belos_TpetraAdapter_MP_Vector.hpp>

// Kokkos
#include <Kokkos_DefaultNode.hpp>

// MueLu
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <MueLu_TpetraOperator_fwd.hpp>

// Teuchos
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>

// Tpetra
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>

// STL header files
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <exception>
#include <fstream>
#include <limits>
#include <random>
#include <string>

// Other header files
#include <mpi.h>

int main(int argc, char** argv) {
    
    // Avoid excessive typing
    using std::cout;
    using std::endl;

    // Check that the correct number of arguments have been passed
    if (argc < 2) {
        cout << "Usage ./mtx_optimal <matrix market file name>" << endl;
        return EXIT_FAILURE;
    }
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    
    {
        // Build communicator
        Teuchos::RCP<const Teuchos::Comm<int>> t_comm(new Teuchos::MpiComm<int> (comm));
        
        // Define scalar, local ordinal, global ordinal, and node type
        typedef double ST; 
        typedef int    LO; 
        typedef int    GO; 
        typedef KokkosClassic::DefaultNode::DefaultNodeType NT; 
        
        // Define matrix, multivector, row map, column map, and muelu operator
        // types from the previously defined scalar, local ordinal, global ordinal,
        // and node type
        typedef Tpetra::CrsMatrix<ST, LO, GO, NT> mtx_t;
        typedef Tpetra::Operator<ST, LO, GO, NT> tpetra_op_t;
        typedef Tpetra::MultiVector<ST, LO, GO, NT> mv_t;
        typedef Tpetra::Map<LO, GO, NT> row_map_t;
        typedef Tpetra::Map<LO, GO, NT> col_map_t;
        typedef MueLu::TpetraOperator<ST, LO, GO, NT> muelu_op_t;

        // Get name of matrix file and name of matrix
        std::string mtx_filename = argv[1];
        std::string mtx_name     = mtx_filename.substr(14, mtx_filename.length() - 18);
        
        // Label file
        std::string label_filename = "data/labels/" + mtx_name + ".csv";
        std::string ps_filename    = "data/ps/"     + mtx_name + ".csv"; 
        
        // Load the .mtx file to an array
        Teuchos::RCP<mtx_t> A;
        try {
            A = Tpetra::MatrixMarket::Reader<mtx_t>::readSparseFile(mtx_filename, t_comm);
        } 
        catch (std::exception e) {
            cout << "Error reading file: " << mtx_filename << endl;
            return EXIT_FAILURE;
        }


        // Number of particles
        const int n_particles = 20;

        // Number of parameters
        const int n_params = 5;

        // Define particle struct and array of particles for particle swarm
        typedef struct {
            double pos[n_params];
            double vel[n_params];
            double cost;
            double best_pos[n_params];
            double best_cost;
        } particle_t;

        particle_t particles[n_particles];

        // Define the cost function as a capturing lambda
        auto cost_fn = [&](particle_t& p) {
           
            double drop_tolerance = -1.0;
            double relaxation_damping = -1.0;
            double sa_damping = -1.0;
            int coarse_max_size = -1;
            int relaxation_sweeps = -1;

            if (t_comm->getRank() == 0) {            
                // Take normalized params to MueLu parameters
                drop_tolerance     = std::exp(p.pos[0] * 8.0 - 10.0);
                relaxation_damping = p.pos[1] * 1.6 + 0.2; 
                sa_damping         = p.pos[2] * 1.6 + 0.2; 
                coarse_max_size    = (int) (p.pos[3] * 9500.0 + 500.0);
                relaxation_sweeps  = (int) (p.pos[4] * 99.0   + 1.0);
            }

            // Use the parameters select on rank 0 on all ranks
            MPI_Bcast(&drop_tolerance,     1, MPI_DOUBLE, 0, comm);
            MPI_Bcast(&relaxation_damping, 1, MPI_DOUBLE, 0, comm);
            MPI_Bcast(&sa_damping,         1, MPI_DOUBLE, 0, comm);
            MPI_Bcast(&coarse_max_size,    1, MPI_INT,    0, comm);
            MPI_Bcast(&relaxation_sweeps,  1, MPI_INT,    0, comm);

            // Default parameters
            std::string multigrid_algorithm  = "sa";
            std::string verbosity            = "none";

            // Parameter lists
            Teuchos::ParameterList param_list;
            Teuchos::ParameterList smooth_list;
            
            // Set default parameters
            param_list.get("multigrid algorithm", multigrid_algorithm);
            param_list.get("smoother: type", "RELAXATION");
            param_list.get("verbosity", verbosity);
            
            // Set random parameters            
            param_list.get("aggregation: drop tol", drop_tolerance);
            param_list.get("sa: damping factor", sa_damping);
            param_list.get("coarse: max size", coarse_max_size);
            param_list.get("max levels", 20);
            
            smooth_list.get("relaxation: type", "Symmetric Gauss-Seidel");
            smooth_list.get("relaxation: damping factor", relaxation_damping);
            smooth_list.get("relaxation: sweeps", relaxation_sweeps);
            param_list.get("smoother: params", smooth_list);
            
            if (t_comm->getRank() == 0) {
                cout << "Solving system with params:" << endl;
                cout << "\tDrop tolerance: "     << drop_tolerance     << endl;
                cout << "\tRelaxation Damping: " << relaxation_damping << endl;
                cout << "\tSA Damping: "         << sa_damping         << endl;
                cout << "\tCoarse Max Size: "    << coarse_max_size    << endl;
                cout << "\tRelaxation Sweeps: "  << relaxation_sweeps  << endl;
            }

            // Make sure all ranks are on this loop iteration before
            // constructing the preconditioner
            t_comm->barrier();

            // Start the clock 
            double t_start = MPI_Wtime();

            // Create the MueLu preconditioner
            Teuchos::RCP<muelu_op_t> M = MueLu::CreateTpetraPreconditioner(A, param_list);
            
            // Get the domain map of the read matrix
            Teuchos::RCP<const row_map_t> map = A->getDomainMap();

            // Create a RHS and an initial guess
            int n_rhs = 1;
            
            // Create multivectors representing solution and initial guess
            Teuchos::RCP<mv_t> b;
            Teuchos::RCP<mv_t> x;
            x = Teuchos::rcp(new mv_t(map, n_rhs));
            b = Teuchos::rcp(new mv_t(map, n_rhs));
            Belos::MultiVecTraits<ST, mv_t>::MvRandom(*x);
            Belos::OperatorTraits<ST, mv_t, tpetra_op_t>::Apply(*A, *x, *b);
            Belos::MultiVecTraits<ST, mv_t>::MvInit(*x, 0.0);

            // Create a linear problem
            Teuchos::RCP<Belos::LinearProblem<ST, mv_t, tpetra_op_t>> problem;
            problem = Teuchos::rcp(new Belos::LinearProblem<ST, mv_t, tpetra_op_t>(A, x, b));
            
            // Set the linear problem to the left preconditioner
            problem->setLeftPrec(M);
            bool set = problem->setProblem();

            // Parameters for Belos solver
            Teuchos::ParameterList belos_list_raw;
            belos_list_raw.set("Block Size", 1);
            belos_list_raw.set( "Use Single Reduction", true );
            belos_list_raw.set("Maximum Iterations", 1000);
            belos_list_raw.set("Convergence Tolerance", 1e-10);
            belos_list_raw.set("Output Frequency", 1 );
            belos_list_raw.set("Verbosity", Belos::TimingDetails + Belos::FinalSummary);
            Teuchos::RCP<Teuchos::ParameterList> belos_list = Teuchos::rcp(&belos_list_raw, false);

            // Create Belos solver
            Belos::BlockCGSolMgr<ST, mv_t, tpetra_op_t, true> solver(problem, belos_list);

            // Solve system
            Belos::ReturnType ret = solver.solve();
            bool success;
            if (ret != Belos::Converged) {
                success = false;
                std::cout << std::endl << "ERROR:  Belos did not converge!" << std::endl;
            } else {
                success = true;
                std::cout << std::endl << "SUCCESS:  Belos converged!" << std::endl;
            }

            // End the clock
            double t_end = MPI_Wtime();

            // Compute elapsed time. If it was better, we update the best
            // params and best cost
            double elapsed = t_end - t_start;

            if (!success) elapsed = 1e6;
            
            if (t_comm->getRank() == 0) {
                cout << "Took " << elapsed << " seconds to solve" << endl;
            }

            // Return the elapsed time
            return elapsed;
        }; 

        // Random number generators
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> d1(0.0, 1.0);
        std::uniform_real_distribution<double> d2(-1.0, 1.0);

        // Perform particle swarm
        const int n_steps  = 50; 
        const double tol   = 1e-4;
        const double omega = 0.3;
        const double phi_g = 0.3;
        const double phi_p = 0.3;

        // Best position and cost
        double global_best_pos[n_params];
        double global_best_cost = std::numeric_limits<double>::max();

        // Open files on rank 0
        std::ofstream ps_history_fs;
        std::ofstream label_fs;
        if (t_comm->getRank() == 0) {
            ps_history_fs = std::ofstream(ps_filename, std::ios::out | std::ios::trunc);

            ps_history_fs << "Step, Particle";
            for (int i = 0; i < n_params; i++) {
                ps_history_fs << ", Pos" << i;
            }
            for (int i = 0; i < n_params; i++) {
                ps_history_fs << ", Vel" << i;
            }
            ps_history_fs << ", Cost" << endl;

        }

        // Initialize particles
        for (int i = 0; i < n_particles; i++) {
            for (int j = 0; j < n_params; j++) {
                particles[i].pos[j] = d1(gen);
                particles[i].vel[j] = d2(gen);
                particles[i].best_pos[j] = particles[i].pos[j];
            }
            particles[i].cost = cost_fn(particles[i]);
            particles[i].best_cost = particles[i].cost;
        }

        for (int i = 0; i < n_steps; i++) {
            
            // Do particle positon update
            //
            // Calculate global best position
            for (int j = 0; j < n_particles; j++) {
                if (particles[j].best_cost < global_best_cost) {
                    global_best_cost = particles[j].best_cost;
                    if (t_comm->getRank() == 0) {
                        label_fs = std::ofstream(label_filename, std::ios::out | std::ios::trunc);
                        label_fs << "Drop Tolerance, Relaxation Damping, SA Damping, Coarse Max Size, Relaxation Sweeps" << endl;
                    }
                    for (int k = 0; k < n_params; k++) {
                        global_best_pos[k] = particles[j].best_pos[k];
                        if (t_comm->getRank() == 0) {
                            label_fs << global_best_pos[k];
                            if (k != n_params - 1) label_fs << " ,";
                            if (k == n_params - 1) label_fs << "\n";
                        }
                    }
                }
            }

            // Update particle positions
            for (int j = 0; j < n_particles; j++) {
                if (t_comm->getRank() == 0) {
                    ps_history_fs << i << ", " << j << ", ";
                }
                for (int k = 0; k < n_params; k++) {
                    double r_g = d1(gen);
                    double r_p = d1(gen);
                    particles[j].vel[k] = omega * particles[j].vel[k] +
                                          r_g * phi_g * (global_best_pos[k] - particles[j].pos[k]) +
                                          r_p * phi_p * (particles[j].best_pos[k] - particles[j].pos[k]);
                    particles[j].pos[k] += particles[j].vel[k];
                    if (particles[j].pos[k] >= 1.0) particles[j].pos[k] = 0.999;
                    if (particles[j].pos[k] <  0.0) particles[j].pos[k] = 0.0;

                }
                for (int k = 0; k < n_params; k++) {
                    if (t_comm->getRank() == 0) {
                        ps_history_fs << particles[j].pos[k] << ", ";
                    }
                }
                for (int k = 0; k < n_params; k++) {
                    if (t_comm->getRank() == 0) {
                        ps_history_fs << particles[j].vel[k] << ", ";
                    }
                }
                ps_history_fs << particles[j].cost << endl;

                particles[j].cost = cost_fn(particles[j]); 
                if (particles[j].cost < particles[j].best_cost) {
                    particles[j].best_cost = particles[j].cost;
                    for (int k = 0; k < n_params; k++) {
                        particles[j].best_pos[k] = particles[j].pos[k];
                    }
                }
            }
        } // end ps loop
        
        if (t_comm->getRank() == 0) {
            ps_history_fs.close();
        }

    } // end tpetra scope 

    // Finalize MPI
    MPI_Finalize();

    return EXIT_SUCCESS;
}
