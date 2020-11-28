#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdio>
#include <cassert>
#include <armadillo>

using namespace std;
using namespace arma;

//-------------------------------------------------------------------
//
// THIS CODE EVALUATES THE STABILITY OF THE ZERO RACEMIC FIXED POINT.
//
//
// Execution : ./a.out nc_val a0_val
// 
//-------------------------------------------------------------------


#define tau 1

#define sigma 2e-4      // standard deviation of the rate constant distribution
#define mu 1e-4         // mean value of the rate constant distribution




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

/* checks if the trivial racemic fixed point is stable or not. Return 0 if stable, 1 if unstable. */
int dynamics(int len, double* eigReal){
    for (int i = 0; i < len; i++){
        if (eigReal[i] > 0){
            return 1;
        }
    }
    return 0;
}

/* finds the maximum of an 1D-array */
int max(double *tab, int size){
    int imax = 0;
    for (int i = 1; i<size; i++){
        if (tab[imax] < tab[i]){
            imax = i;
        }
    }
    return imax;
}

/* Computes the general Jacobian of the system of 2+2*nc equations */
void JacobEig(double a0, int nc, double***kp, double**km){
    int i, j, k, m, n;
    double s1, s2, s3;
    int size = 2+2*nc;
    
    double * eigReal = (double*)malloc(size*sizeof(double));
    
    mat Jacob(size,size);
    
    Jacob(0,0) = -1/tau;
    Jacob(0,size-1) = 0.;
    
    Jacob(size-1,0) = 0.;
    Jacob(size-1,size-1) = -1/tau;
    
    for (n=0; n<nc; n++){
        s1 = 0.;
        for (j = 0; j<nc; j++){
            for (k=j; k<nc; k++){
                s1 += kp[n][j][k];
            }
        }
        Jacob(0,1+n) = - a0 * s1;
        Jacob(0,1+nc+n) = - a0 * s1;
        
        Jacob(size-1, 1+n) = 0.;
        Jacob(size-1,1+nc+n) = 0.;
    }
    
    for (m=0; m<nc; m++){
        Jacob(1+m,0) = 0.;
        Jacob(1+nc+m,0) = 0.;
        for (n=0; n<nc; n++){
            s1 = 0.;
            s2 = 0.;
            s3 = 0.;
            for (j = m; j<nc; j++){
                s1 += kp[n][m][j];
            }
            for (j=0; j<=m; j++){
                s2 += kp[n][j][m];
            }
            Jacob(1+m,1+n) = a0*(s1+s2);
            Jacob(1+m,1+nc+n) = 0.;
        
            Jacob(1+nc+m,1+n) = 0.;
            Jacob(1+nc+m,1+nc+n) = a0*(s1+s2);
        
            if (m==n){
                for (i=0;i<nc;i++){
                    for (j=i;j<nc;j++){
                        s3+=kp[m][i][j];
                    }
                }
            
                Jacob(1+m,1+n) += -a0*s3 - 1/tau;
                Jacob(1+nc+m,1+nc+n) += -a0*s3 - 1/tau;
            }
        }
    }
    
    cx_vec eigval = eig_gen(Jacob);
    
    for (i=0; i<size; i++){
        eigReal[i] = real(eigval(i));
    }
    
    fstream data;

    data.open("stability.res", ios::out);
    data << dynamics(2+2*nc, eigReal) << endl;
    data.close();
    
    fstream vp;
    vp.open("vp.res", ios::out);
    for (i = 0; i<2*nc+2; i++){
        vp << real(eigval(i)) << " " << imag(eigval(i)) << endl;
    }
    vp.close();
    
}


/* computes the jacobian of size nc*nc of the enantiomeric excess dynamics (see eq. 2) */
void jacobdx(double a0, int nc, double***kp, double**km){
    int i, j, m, n;
    double s1, s2, s3;
    
    double * eigReal = (double*)malloc(nc*sizeof(double));
    
    mat Jacob(nc,nc);
    
    for (m=0; m<nc; m++){
        for (n=0; n<nc; n++){
            s1=0;
            s2=0;
            s3=0;
            for (i=m; i<nc; i++){
                s1+=kp[n][m][i];
            }
            for (i=0; i<=m; i++){
                s2+=kp[n][i][m];
            }
            Jacob(m,n) = a0*(s1+s2);
            
            if (m==n){
                for (i=0; i<nc; i++){
                    for (j=i; j<nc; j++){
                        s3+=kp[n][i][j];
                    }
                }
                Jacob(m,n) += -a0*s3 - 1/tau;
            
            }
        }
    }

    cx_vec eigval = eig_gen(Jacob);
    
    for (i=0; i<nc; i++){
        eigReal[i] = real(eigval(i));
    }
    
    fstream data;

    data.open("stability.res", ios::out);
    data << dynamics(nc, eigReal) << endl;      // write 0 if the racemic trivial state is stable and 1 if not in the stability.res file
    data.close();
    
    fstream vp;
    vp.open("vp.res", ios::out);
    for (i = 0; i<nc; i++){
        vp << real(eigval(i)) << " " << imag(eigval(i)) << endl;    // write eigenvalues in vp.res file
    }
    vp.close();
}


int main(int argc, char* argv[]){
    
    assert(argc == 3 && "Error : please give exactly two arguments to the executable.");
    
    int i,j;
    
    int nc = atoi(argv[1]);
    double a0 = atof(argv[2]);
    
    double*** kp = (double***)malloc(nc*sizeof(double**));  // Tensor of k+_{ijk}
    double** km = (double**)malloc(nc*sizeof(double*));     // Matrix of kt-_{ij}
    
    for (i=0; i<nc; i++) {                                  // Final initializations of k+_{ijk} and kt-_{ij}
        kp[i] = (double**)malloc(nc*sizeof(double*));
        km[i] = (double*)malloc(nc*sizeof(double));
        for (j=0; j<nc; j++) {
            kp[i][j] = (double*)malloc(nc*sizeof(double));
        }
    }
    
    srand48(time(NULL));
    
    random_k(nc, kp, km);           // set random rate constants
//     init_homo_k(nc, kp, km);
    jacobdx(a0, nc, kp, km);        // we use the jacobian described by eq. 2
    free(kp); free(km);
    
    return 0;
}
