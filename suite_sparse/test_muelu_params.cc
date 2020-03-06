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
    if (argc < 3) {
        cout << "Usage ./mtx_optimal <matrix market file name> <MueLu parameter file name>" << endl;
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
        std::string xml_filename = argv[2];

        // Load the .mtx file to an array
        Teuchos::RCP<mtx_t> A;
        try {
            A = Tpetra::MatrixMarket::Reader<mtx_t>::readSparseFile(mtx_filename, t_comm);
        } 
        catch (std::exception e) {
            cout << "Error reading file: " << mtx_filename << endl;
            return EXIT_FAILURE;
        }

        // Start the clock 
        double t_start = MPI_Wtime();

        // Create the MueLu preconditioner
        Teuchos::RCP<muelu_op_t> M;
        M = MueLu::CreateTpetraPreconditioner(A, xml_filename);
        
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
        belos_list_raw.set("Maximum Iterations", 10000);
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
        // params and best reward
        double elapsed = t_end - t_start;

        cout << "Took " << elapsed << " seconds to solve matrix " << mtx_filename << endl;
        cout << "Took " << solver.getNumIters() << " CG iterations to solve matrix " << mtx_filename << endl;

    }        

    // Finalize MPI
    MPI_Finalize();

    return EXIT_SUCCESS;
}
