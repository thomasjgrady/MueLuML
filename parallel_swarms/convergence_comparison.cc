// convergence_comparison.cc
//
// This Program solves Poisson's equation on a two dimensional L-shaped domain
// with homogeneous Dirichlet boundary conditions and constant nonzero load
// vector using Deal.II's conjugate gradient solver with and without 
// AMG preconditioning applied. It performs 3 runs with and without AMG,
// each with various levels of added noise.
//      1. No added noise
//      2. Noise added to load vector
//      3. Noice added to load vector and boundary conditions
//
// The resulting residual data is then output to six csv files, and the
// computed solutions are output to six vtk files for further processing.
//
// To compile the program using cmake/make, run "cmake ." in this directory
// followed by "make convergence_comparison"

// Deal.II Includes
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/trilinos_epetra_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
#include <cstdlib>
# include <MueLu_CreateEpetraPreconditioner.hpp>
# include <ml_MultiLevelPreconditioner.h>

// STL Includes
#include <fstream>
#include <iostream>
#include <random>
#include <string>

// Use the dealii namespace to prevent overly verbose code
using namespace dealii;

// Randomizatiomns
bool use_rhs_randomizations = false;
bool use_boundary_randomizations = false;

// Preconditioner options
bool use_amg_preconditioning = false;

// Output file names
std::string csv_filename = "";
std::string vtk_filename = "";

// Distributions
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution dis{-0.1, 0.1};

// Taken from deal.ii step 4 tutorial program
template <int dim>
class PoissonProblem
{
public:
  PoissonProblem ();
  void run ();
private:
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void solve ();
  void output_results () const;
  Triangulation<dim>   triangulation;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       solution;
  Vector<double>       system_rhs;
};

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide () : Function<dim>() {}
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};
template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues () : Function<dim>() {}
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};
template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
    double result = 1.0;
    if (use_rhs_randomizations) {
        result += dis(gen);    
    }
    return result;

}
template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
    double result = 0.0;
    if (use_boundary_randomizations) {
        result += dis(gen);
    }

    return result;
}
template <int dim>
PoissonProblem<dim>::PoissonProblem ()
  :
  fe (1),
  dof_handler (triangulation)
{}
template <int dim>
void PoissonProblem<dim>::make_grid ()
{
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global(8);
  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: "
            << triangulation.n_cells()
            << std::endl;
}
template <int dim>
void PoissonProblem<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit (sparsity_pattern);
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}
template <int dim>
void PoissonProblem<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(2);
  const RightHandSide<dim> right_hand_side;
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      cell_matrix = 0;
      cell_rhs = 0;
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                                   fe_values.shape_grad (j, q_index) *
                                   fe_values.JxW (q_index));
            cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                            right_hand_side.value (fe_values.quadrature_point (q_index)) *
                            fe_values.JxW (q_index));
          }
      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            BoundaryValues<dim>(),
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}
template <int dim>
void PoissonProblem<dim>::solve ()
{

  // Create a solver control
  SolverControl solver_control(1000, 1e-12);

  // Enable history data
  solver_control.enable_history_data();

  // Create a conjugate gradient solver
  SolverCG<> solver(solver_control);

  // If amg preconditing
  if (use_amg_preconditioning) {

    // Get matrix in trilinos form
    TrilinosWrappers::SparseMatrix trilinos_system_matrix;
    trilinos_system_matrix.reinit(system_matrix);

    // Create amg muelu preconditioner
    TrilinosWrappers::PreconditionAMGMueLu preconditioner;
    Teuchos::ParameterList parameterList;
    preconditioner.initialize(trilinos_system_matrix, parameterList);
    
    // Solve the system    
    solver.solve(system_matrix, solution, system_rhs, preconditioner);

  }

  else {
    // Solve the system without preconditioning
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  }

    // Get residual history
    std::vector<double> residuals = solver_control.get_history_data();

    // Write residual history to csv file
    std::ofstream cg_residual_file(csv_filename);
    cg_residual_file << "step,residual\n";
    for (int i = 0; i < residuals.size(); i++) {
        //std::cout << "Residual at step " << i << ": " << residuals.at(i) << std::endl;
        cg_residual_file << i << "," << residuals.at(i) << "\n";
    }
    cg_residual_file.close();

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;
}
template <int dim>
void PoissonProblem<dim>::output_results () const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();
  std::ofstream output (vtk_filename);
  data_out.write_vtk (output);
}
template <int dim>
void PoissonProblem<dim>::run ()
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
  make_grid();
  setup_system ();
  assemble_system ();
  solve ();
  output_results ();
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    // No randomizations, no preconditioning 
    PoissonProblem<2> poisson_problem_2d_1;
    use_rhs_randomizations = false;
    use_boundary_randomizations = false;
    use_amg_preconditioning = false;
    csv_filename = "poisson-1.csv";
    vtk_filename = "poisson-1.vtk";
    poisson_problem_2d_1.run();

    // No randomizations, with preconditioning 
    PoissonProblem<2> poisson_problem_2d_2;
    use_rhs_randomizations = false;
    use_boundary_randomizations = false;
    use_amg_preconditioning = true;
    csv_filename = "poisson-2.csv";
    vtk_filename = "poisson-2.vtk";
    poisson_problem_2d_2.run();

    // RHS randomizations, no preconditioning 
    PoissonProblem<2> poisson_problem_2d_3;
    use_rhs_randomizations = true;
    use_boundary_randomizations = false;
    use_amg_preconditioning = false;
    csv_filename = "poisson-3.csv";
    vtk_filename = "poisson-3.vtk";
    poisson_problem_2d_3.run();

    // RHS randomizations, with preconditioning 
    PoissonProblem<2> poisson_problem_2d_4;
    use_rhs_randomizations = true;
    use_boundary_randomizations = false;
    use_amg_preconditioning = true;
    csv_filename = "poisson-4.csv";
    vtk_filename = "poisson-4.vtk";
    poisson_problem_2d_4.run();

    // RHS and boundary randomizations, no preconditioning 
    PoissonProblem<2> poisson_problem_2d_5;
    use_rhs_randomizations = true;
    use_boundary_randomizations = true;
    use_amg_preconditioning = false;
    csv_filename = "poisson-5.csv";
    vtk_filename = "poisson-5.vtk";
    poisson_problem_2d_5.run();

    // RHS randomizations, with preconditioning 
    PoissonProblem<2> poisson_problem_2d_6;
    use_rhs_randomizations = true;
    use_boundary_randomizations = true;
    use_amg_preconditioning = true;
    csv_filename = "poisson-6.csv";
    vtk_filename = "poisson-6.vtk";
    poisson_problem_2d_6.run();
    MPI_Finalize();
}
