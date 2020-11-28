#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdio>
#include <armadillo>
#include <cassert>

using namespace arma;
using namespace std;

/* This code simulates generalized Frank's model with a Runge-Kutta of order 2 scheme */

#define tau 1.  // set tau to 1 for numerical simulations

#define h 0.1   // time step for differential solving

// Concentrations
#define at0 0.  // no unactivated achiral species at initial time

#define c0 2.   // initial concentrations of chiral species
#define eps 1e-2    // bias in initial concentrations of chiral species
#define eps_x 1e-10 // "accuracy" of the steady state

// Kinetic rates
#define mu 1e-4 // mean of the rate constant distribution
#define sigma 2e-4  // standard deviation of the rate constant distribution


/* This function computes the derivative of A (concentration of activated achiral specie) */
double dotA(double* x, double*** kp, int nc, double a0){
    int i,j,k;	// Some increments
    double s1 = 0., s2 = 0.;

    for (i=0; i<nc; i++){
        for (k=0; k<nc; k++){
            for (j=0; j<=k; j++){
		    s1 = s1 + kp[i][j][k]*x[0]*x[2+i];
		    s2 = s2 + kp[i][j][k]*x[0]*x[2+nc+i];
            }
        }
    }
    
    return -s1-s2+(a0-x[0])/tau;
}

/* This function computes the derivative of D (concentration of D enantiomers) */
double dotD(double* x, double*** kp, double** km, int m, int nc){
    int i,j;
    double s1 = 0., s2 = 0., s3 = 0., s4 = 0.;

    for (j=0; j<nc; j++){
        for (i=0; i<=j; i++){
            s1 = s1 + kp[m][i][j]*x[0]*x[2+m];
        }
    }

    for (i=0; i<nc; i++){
        for (j=m; j<nc; j++){
            s2 = s2 + kp[i][m][j]*x[0]*x[2+i];
        }
    }

    for (i=0; i<nc; i++){
        for (j=0; j<=m; j++){
            s3 = s3 + kp[i][j][m]*x[0]*x[2+i];
        }
    }

    for (i=0; i<nc; i++){
        s4 = s4 + km[m][i]*x[2+m]*x[2+nc+i];
    }

    return -s1+s2+s3-s4-x[2+m]/tau;    
}

/* This function computes the derivative of L (concentration of L enantiomers) */
double dotL(double* x, double*** kp, double** km, int m, int nc){
    int i,j;
    double s1 = 0., s2 = 0., s3 = 0., s4 = 0.;

    for (j=0; j<nc; j++){
        for (i=0; i<=j; i++){
            s1 = s1 + kp[m][i][j]*x[0]*x[2+nc+m];
        }
    }

    for (i=0; i<nc; i++){
        for (j=m; j<nc; j++){
            s2 = s2 + kp[i][m][j]*x[0]*x[2+nc+i];
        }
    }

    for (i=0; i<nc; i++){
        for (j=0; j<=m; j++){
            s3 = s3 + kp[i][j][m]*x[0]*x[2+nc+i];
        }
    }

    for (i=0; i<nc; i++){
        s4 = s4 + km[i][m]*x[2+i]*x[2+nc+m];
    }

    return -s1+s2+s3-s4-x[2+nc+m]/tau;
}

/* This function computes the derivative of At (concentration of unactivated achiral specie) */
double dotAt(double* x, double** km, int nc){
    int i,j;
    double s1 = 0.;

    for (i=0; i<nc; i++){
        for (j=0; j<nc; j++){
            s1 = s1 + km[i][j]*x[2+i]*x[2+nc+j];
        }
    }
    
    return 2*s1-x[1]/tau;
}

/* This function computes the derivaties of the system of concentrations */
void system(double* x, double* xp, double*** kp, double** km, int nc, double a0) {

    xp[0] = dotA(x, kp, nc, a0);    // Activated achiral specie
    xp[1] = dotAt(x, km, nc);   // Unactivated achiral specie
    
    for (int i=0; i<nc; i++){
        xp[2+i] = dotD(x, kp, km, i, nc);       // D enantiomers
        xp[2+nc+i] = dotL(x, kp, km, i, nc);    // L enantiomers
    }
}

/* Perform a RK2 (Runge-Kutta of order 2) step */
void rk2(double* x, double*** kp, double ** km, int nc, double a0){
    double a = 1./3, b = 2./3, be = 3./4;   // RK2 parameters
    int i;
    int size = 2+2*nc;
    
    double* xp = (double*)malloc(size*sizeof(double));  // Derivatives vector
    
    double* x_shift = (double*)malloc(size*sizeof(double)); // Vector containing new xp determined with k1
    
    double* k1 = (double*)malloc(size*sizeof(double));  // Vector containing k1s
    double* k2 = (double*)malloc(size*sizeof(double));  // Vector containing k2s
    
    system(x, xp, kp, km, nc, a0);
    
    for (i=0; i<size; i++){
        k1[i] = h*xp[i];
        x_shift[i] = x[i] + be*k1[i];
    }
    
    system(x_shift, xp, kp, km, nc, a0);
    
    for (i=0; i<size; i++){
        k2[i] = h*xp[i];
        x[i] = x[i] + a*k1[i] + b*k2[i];
    }
    
    free(k1); free(k2); free(xp); free(x_shift);
}

/* This function checks if the change in concentration exceed a certain value eps_x. If yes, the simulation goes on. If not, it stops. */
int continueorstop(double* x, double* x_temp, int size){
    for (int i = 0; i < size; i++){
        if (abs(x[i] - x_temp[i]) > eps_x){
            return 1;
        }
    }
    return 0;
}

/* This function simulates the frank's model over time */
void simul(double* x, double*** kp, double** km, int nc, double a0){
    int size = 2+2*nc;
    int j=0;
    double t = 0;
    double* x_temp = (double*)malloc(size*sizeof(double));
    fstream data;
    data.open("simulation.res", ios::out);
    
    for ( int i = 0; i < size ; i++){
        x_temp[i] = 0;
    }
    
    while(continueorstop(x, x_temp, size)){
        
        for (int i = 0; i < size; i++){
            x_temp[i] = x[i];
        }
        
        if ((j%1)==0){
            data << t;
            for (int i=0; i<size; i++){
                data << " " << x[i];
            }
            data << "\n";
        }

        t += h;
        rk2(x, kp, km, nc, a0);
        
        for (int i = 0; i < size; i++){
            if ((x[i]<0) or (isnan(x[i]))){
                cout << "A concentration takes a forbidden value at time t = " << t << endl;
                exit(1);
            }
            j++;
        }
        

    }
     data.close();
}

/* This function generates random values for kinetic rates according to a log-normal distribution or read them from an already existing file */
void random_k(int nc, double*** kp, double** km){
    int i,j,k;
    double ktemp;
    double mean, ecart;
    fstream fichkp;
    fstream fichkm;
    string randomkp = "random_kp_" + to_string(nc) + ".txt";
    string randomkm = "random_km_" + to_string(nc) + ".txt";
    fichkp.open(randomkp);
    
    mean = log(mu) - 0.5*log(1+(sigma*sigma)/(mu*mu));
    ecart = sqrt(log(1+(sigma*sigma)/(mu*mu)));
    
    if (not fichkp){
        fichkp.open(randomkp, ios::out);
        fichkm.open(randomkm, ios::out);
        
        for (i=0; i<nc; i++){
            for (j=0; j<nc; j++){
                for (k=0; k<nc; k++){
                    ktemp = exp(ecart*sqrt(-2*log(drand48()))*cos(2*M_PI*drand48())+mean);
                    kp[i][j][k] = ktemp;
                    fichkp << ktemp << endl;
                }
            }
        }

        for (i=0; i<nc; i++){
            for (j=0; j<=i; j++){
                ktemp = exp(ecart*sqrt(-2*log(drand48()))*cos(2*M_PI*drand48())+mean);
                km[i][j] = ktemp;
                km[j][i] = ktemp;
                fichkm << ktemp << endl;
                }
            }
        fichkp.close();
        fichkm.close();
    } 

    else {
        fichkp.close();
        fichkp.open(randomkp, ios::in);
        fichkm.open(randomkm, ios::in);
        
        for (i=0; i<nc; i++){
            for (j=0; j<nc; j++){
                for (k=0; k<nc; k++){
                    fichkp >> kp[i][j][k];
                }
            }
        }
        for (i=0; i<nc; i++){
            for (j=0; j<=i; j++){
                fichkm >> ktemp;
                km[i][j] = ktemp;
                km[j][i] = ktemp;
            }
        }
        fichkp.close();
        fichkm.close();
    }
}

/* This function generates homogeneous rate constants */
void init_homo_k(int nc, double*** kp, double ** km){
    int i,j,k;
    
    for (i=0; i<nc; i++){
        for (j=0; j<nc; j++){
            for (k=0; k<nc; k++){
                kp[i][j][k] = 1e-4;
            }
            km[i][j] = 1e-4;
        }
    }
}

/* This function initiates the intial concentrations with a small perturbation */
void init_concentrations(double* x, int nc, double a0){
    x[0] = a0;             // Initialization of concentrations
    x[1] = at0;
    for (int i=0; i<nc; i++){
        x[2+i] = c0+eps;    // Introduction of infinitesimal deviation (eps) from racemic subspace
        x[2+nc+i] = c0-eps;
    }
}

    
int main(int argc, char *argv[]) {
    
    assert(argc == 3 && "Error : please give exactly two arguments to the executable.");
    
    int nc = atoi(argv[1]);
    double a0 = atof(argv[2]);
    
    double* x = (double*)malloc((2+2*nc)*sizeof(double));   // Vector of concentrations
    double*** kp = (double***)malloc(nc*sizeof(double**));  // Tensor of k+_{ijk}
    double** km = (double**)malloc(nc*sizeof(double*));     // Matrix of kt-_{ij}
    
    for (int i=0; i<nc; i++) {                                  // Final initializations of k+_{ijk} and kt-_{ij}
        kp[i] = (double**)malloc(nc*sizeof(double*));
        km[i] = (double*)malloc(nc*sizeof(double));
        for (int j=0; j<nc; j++) {
            kp[i][j] = (double*)malloc(nc*sizeof(double));
        }
    }
        
    srand48(time(NULL));
    
    init_concentrations(x, nc, a0);
    random_k(nc, kp, km);           // If we want random rate constants
//    init_homo_k(nc, kp, km);      // If we want equal rate constants
    simul(x, kp, km, nc, a0);
    
    free(x); free(kp); free(km);
    
    return 0;
}
