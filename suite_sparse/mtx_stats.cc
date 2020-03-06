// trilinos includes

// Kokkos
#include <Kokkos_DefaultNode.hpp>

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
#include <TpetraExt_MatrixMatrix_def.hpp>

// STL header files
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <stdio.h>
#include <ctime>

//other
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
		//declare constant sampling rows
		const int num_samples = 1601;

	    // Define scalar, local ordinal, global ordinal, and node type
	    typedef double ST;
	    typedef int    LO;
	    typedef int    GO;
	    typedef KokkosClassic::DefaultNode::DefaultNodeType NT;

	    // Define matrix, multivector, row map, column map, and muelu operator
	    // types from the previously defined scalar, local ordinal, global ordinal,
	    // and node type
	    typedef Tpetra::CrsMatrix<ST, LO, GO, NT> mtx_t;
		typedef Tpetra::Vector<ST, LO, GO, NT> vec_t;
	    typedef Tpetra::Operator<ST, LO, GO, NT> tpetra_op_t;
	    typedef Tpetra::MultiVector<ST, LO, GO, NT> mv_t;
	    typedef Tpetra::Map<LO, GO, NT> row_map_t;
	    typedef Tpetra::Map<LO, GO, NT> col_map_t;
	    typedef MueLu::TpetraOperator<ST, LO, GO, NT> muelu_op_t;

	    // Build communicator
	    Teuchos::RCP<const Teuchos::Comm<int>> t_comm(new Teuchos::MpiComm<int> (comm));

		//set rank
		int rank = t_comm->getRank();

	    // Get the matrix marker file name from passed args
	    std::string mtx_filename = argv[1];

	    if (rank == 0) {
	        cout << "Reading data from mtx file: " << mtx_filename << endl;
	    }

	    // Read the matrix market file to a Tpetra matrix
	    Teuchos::RCP<mtx_t> A;
	    try {
	        A = Tpetra::MatrixMarket::Reader<mtx_t>::readSparseFile(mtx_filename, t_comm);
	    } catch (...) {
	        cout << "Error reading file: " << mtx_filename << endl;
	    }
        
        // Get the number of rows of A
        GO global_num_rows = A->getGlobalNumRows();
        
        // Initialize random number generators
        srand(time(NULL));
        std::mt19937(gen);
        std::uniform_int_distribution<GO> dis(0, global_num_rows);

        // Open a file stream
        std::ofstream ofs("

        // Loop over and get 

    }

    MPI_Finalize();

    return EXIT_SUCCESS;
} 
        
