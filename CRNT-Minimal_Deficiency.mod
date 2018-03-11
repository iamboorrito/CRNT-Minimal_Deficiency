/*********************************************
 * OPL 12.8.0.0 Model
 * Author: Evan Burton
 * Creation Date: Feb 25, 2018 at 6:31:22 PM
 *********************************************/

// Clocking
/////////////////////////
float temp;
execute{
var before = new Date();
temp = before.getTime();
}
/////////////////////////

// Data declarations
int rows = ...;
int cols = ...;
float Y[1..rows][1..cols] = ...;
float M[1..rows][1..cols] = ...;
float eps = ...;
int s = ...;

// Coordinate pair data structure needed for iteration over
// custom sets of indices
tuple pair {
	int i;
	int j;
};

// This is the set of pairs (i,j) s.t. that i != j
{pair} off_diag = {<i, j> | i,j in 1..cols: i != j};

// Decision variable declarations
dvar float Ab[1..cols][1..cols];
dvar float T[1..rows];
// Should come up with better way of storing/indexing delta since there are
// only n(n-1) values used and n ignored.
dvar int gamma[1..cols][1..cols-s];
dvar float theta[1..cols-s];
dvar float phi[1..cols][1..cols];

// Objective function
minimize (cols - s - sum(k in 1..cols-s) theta[k]);

// Begin constraint declarations
subject to {

/*
	// Heuristic solution
	forall(i in 1..rows-1){
		T[i] == 1;
	}
*/	
	// Y Ab = T M
	forall(i in 1..rows, j in 1..cols){
		sum(k in 1..cols) Y[i][k]*Ab[k][j] - T[i]*M[i][j] == 0;
	}
	
	// Columns of Ab sum to 0
	forall(j in 1..cols){	
		sum(i in 1..cols) Ab[i][j] == 0;
	}
	
	// Constraints on the off diagonal entries of Ab
	forall(<i,j> in off_diag){
			// Ab constraints
		Ab[i][j] >= 0;
		Ab[i][j] <= 1/eps;
	}
	
	// Bound diagonal entries of Ab
	forall(j in 1..cols){	
	
		Ab[j][j] <= 0;
			
	}
	
	// Bounds on T
	forall(i in 1..rows){
		eps <= T[i] <= 1/eps;	
	}

	// Partitioning
	forall(i in 1..cols){
		sum(k in 1..cols-s) gamma[i][k] == 1;
		
		forall(k in 1..cols-s){
			0 <= gamma[i][k] <= 1;
			// Uniqueness
			if(k <= i)
				sum(j in 1..i-1) gamma[j][k] >= sum(L in k+1..cols-s) gamma[i][L];
		}	
	}
	
	forall(k in 1..cols-s){
		sum(i in 1..cols) gamma[i][k] >= eps*theta[k];
		sum(i in 1..cols) gamma[i][k] <= 1/eps*theta[k];
		0 <= theta[k] <= 1;
	}
	
	// Kernal constraints
	forall(<i,j> in off_diag){
		sum(L in 1..i-1) (phi[i][L] - phi[L][i]) == 0; 
		sum(L in i+1..cols) (phi[i][L] - phi[L][i]) == 0; 
		
		forall(k in 1..cols-s){
			phi[i][j] <= 1/eps*(gamma[i][k] - gamma[j][k] + 1);
			
			phi[i][j] >= eps*Ab[i][j];
			phi[i][j] <= 1/eps*Ab[i][j];
		}	
		
	}
}

// Clocking
//////////////////////////
execute{
var after = new Date();
writeln("solving time ~= ", (after.getTime()-temp));
}
//////////////////////////

execute DISPLAY{};
