#define INT int

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef NULL
#define NULL 0
#endif


/* ========================================================================== */
/* === PPparm =============================================================== */
/* ========================================================================== */

typedef struct PPparm_struct
{
    int     use_lambda ; /* T = use input lambda as starting guess */
    int        cholmod ; /* T = use cholmod and update/downdates, F = SSOR */
    int     multilevel ; /* T = use multilevel method, F = single level */
    int        stopabs ; /* T = use absolute stopping criterion, F = relative */
    int   providePPcom ; /* T = memory provided by user in userI (interface) */
    int     freememory ; /* T = free memory before exit, F = keep memory =>
                                Interface structure returned as void pointer */
    double       sigma ; /* penalty for prox lambda term, also chol diag shift*/
    int     ScaleSigma ; /* T = scale sigma by max_ij |a_ij| */
    double sigma_decay ; /* decay factor for sigma in the prox iteration */
    int          nprox ; /* max number of proximal updates */
    double armijo_grow ; /* growth factor for SpaRSA's Armijo step */
    int        narmijo ; /* maximum number of Armijo expansions */
    int            mem ; /* number of function values stored for SpaRSA */
    int        nsparsa ; /* maximum number of SpaRSA iterations */
    double       gamma ; /* parameter in (0, 1) for terminating SpaRSA */
    double         tau ; /* tau, beta, grad_decay, gamma_decay are (continued)*/
    double        beta ; /* parameters associated with formula for (continued)*/
    double  grad_decay ; /* undecided indices -- see default in    (continued)*/
    double gamma_decay ; /* pproj for details */
    int use_coor_ascent; /* T = use coordinate ascent, F = do not use */
    double    coorcost ; /* if coorcost*Annz <= Lnnz, use coor_ascent */
    int      use_ssor0 ; /* T = use ssor0 if appropriate, F = never use ssor0 */
    int      use_ssor1 ; /* T = use ssor1 if appropriate, F = never use ssor1 */
    int     use_sparsa ; /* T = use SpaRSA if appropriate, F=never use SpaRSA */
    int     use_phase1 ; /* T = use phase1, F = skip phase1 */
    double   ssordecay ; /* stop ssor1 when errls <= ssordecay * errdual */
    double    ssorcost ; /* if ssorcost*Annz <= Lnnz, use ssor */
    int        ssormem ; /* number of vectors in ssor memory */
    int     ssormaxits ; /* upper bound on number of ssor iterations */
    double   cutfactor ; /* factor used for binding constraints in ascent */
    double     tolssor ; /* stop ssor when err1 <= err/tol2 */
    double     tolprox ; /* prox update when err <= tol2*norm_l */
    double tolrefactor ; /* refactorization tolerance for Cholesky factor */
    double      phase1 ; /* phase1 iterations = MAX (nrow^{phase1}, 5),
                            phase1 < 0 => no phase1 iterations */
    int     PrintLevel ; /* = 0(none), = 1(final), = 2(post loop), 3(in-loop) */
    double       debug ; /* debug level, 1 = after loop, 2 = in loop*/
    double    checktol ; /* acceptable errors in debug routines */
    double itup ;
} PPparm ;

/* ========================================================================== */
/* === PPstat =============================================================== */
/* ========================================================================== */

typedef struct PPstat_struct
{
    INT             nrow ; /* number of rows in A */
    double       errdual ; /* dual gradient (absolute if Parm->stopabs = TRUE
                                             relative if Parm->stopabs = FALSE
                                             see pproj for details) */
    int         *updowns ; /* number of updates and downdates of each size */
    int     size_updowns ; /* dimension of updowns array */
    int          *solves ; /* size: maxdepth+1, # of solves by level  */
    int         maxdepth ; /* number of levels in the partition tree */
    int             blks ; /* number of blocks in the multilevel partition of A
                              each separator is also counted as a block */
    INT       phase1_its ; /* number of iterations in phase 1 */
    INT  coor_ascent_its ; /* number of coordinate ascent iterations */
    INT        ssor0_its ; /* number of ssor0 iterations */
    INT        ssor1_its ; /* number of ssor1 iterations */
    INT       sparsa_its ; /* number of SpaRSA iterations */
    int            nprox ; /* number of proximal updates */
    INT            coldn ; /* number of rank 1 downdates to Cholesky factor */
    INT            colup ; /* number of rank 1 updates to Cholesky factor */
    INT            rowdn ; /* number of rows dropped from Cholesky factor */
    INT            rowup ; /* number of rows added to Cholesky factor */
    INT coor_ascent_free ; /* number of variables freed in coordinate ascent */
    INT coor_ascent_drop ; /* number of rows dropped in coordinate ascent */
    INT       ssor0_free ; /* number of variables freed in ssor0 */
    INT       ssor0_drop ; /* number of rows dropped in ssor0 */
    INT       ssor1_free ; /* number of variables freed in ssor1 */
    INT       ssor1_drop ; /* number of rows dropped in ssor1 */
    INT       sparsa_col ; /* number of changes in bound constraints in SpaRSA*/
    INT       sparsa_row ; /* number of changes in row constraints in SpaRSA */
    INT sparsa_step_fail ; /* number of failures of Armijo step in SpaRSA */
    INT           nchols ; /* number of Cholesky factorizations */
    INT             lnnz ; /* number of nonzeros in final Cholesky factor */

    /* timing */
    double     partition ; /* compute reordering of rows of A */
    double    initialize ; /* initwork and initlevels, includes partition */
    double        phase1 ; /* phase1 */
    double        sparsa ; /* sparsa */
    double   coor_ascent ; /* coor_ascent */
    double         ssor0 ; /* ssor0 */
    double         ssor1 ; /* ssor1 */
    double          dasa ; /* dasa (includes coor_ascent, ssor0, and ssor1) */
    double     dasa_line ; /* dasa line search */
    double      checkerr ; /* check_error */
    double   prox_update ; /* prox_update */
    double        invert ; /* invert permutation of rows and columns */
    double        modrow ; /* modrow (update L by adding or deleting rows) */
    double        modcol ; /* modcol (rank 1 column updates of L) */
    double          chol ; /* cholmod_analyze, cholmod_factorize */
    double       cholinc ; /* incremental cholmod_rowfac */
    double      dltsolve ; /* dltsolve (back solve) */
    double        lsolve ; /* lsolve (forward solve) */
} PPstat ;

/* prototypes */

int pproj /* return status of solution process:
                0 (convergence tolerance satisfied)
                1 (convergence tolerance not met)
                2 (invalid bound bl [i] > bu [i])
                3 (out of memory)
                4 (ssor0 did not generate a descent direction)
                5 (ssor1 did not generate a descent direction) */
(
    double       *x, /* solution (size ncol, output) */
    double  *lambda, /* multiplier (size nrow, output), if Uparm->use_lambda
                        = TRUE => starting guess on input */
    void     *userI, /* Parm->freememory = F => return PPcom structure
                        Parm->usememory  = T => userI = PPcom structure*/
    double grad_tol, /* relative error tolerance for dual function grad */
    PPparm   *UParm, /* NULL => use default parameters */
    PPstat   *UStat, /* NULL => do not return statistics */
    double       *y, /* project y onto polyhedron (size ncol) */
    INT        nrow, /* number of rows in A */
    INT        ncol, /* number of cols in A */
    INT         *Ap, /* size ncol+1, column pointers */
    INT         *Ai, /* size Ap [ncol], row indices for A in increasing
                        order in each column */
    double      *Ax, /* size Ap [ncol], numerical entries of A */
    double      *lo, /* size n, lower bounds for x */
    double      *hi, /* size n, upper bounds for x */
    double      *Bl, /* size n, lower bounds for b denoted bl above */
    double      *Bu  /* size n, upper bounds for b denoted bu above */
) ;

void pproj_default
(
    PPparm *Parm /* Parameter structure */
) ;

void pproj_print_stat
(
    PPstat   *Stat, /* pointer to statistics structure */
    int freememory  /* T => free Stat->updowns and Stat->solves */
) ;

double pproj_KKTerror /* returns the largest of the primal and dual errors */
(
    double    *errb, /* sup norm error in b (only return if not NULL) */
    double    *errx, /* sup norm error in x (only return if not NULL) */
    double   *absAx, /* sup norm of absAx */
    double       *x, /* computed projection (size ncol) */
    double  *lambda, /* multiplier (size nrow) */
    double       *y, /* y should be projected onto polyhedron (size ncol) */
    INT        nrow, /* number of rows in A */
    INT        ncol, /* number of cols in A */
    int  PrintLevel, /* 0 = no printout, 1 = print constraint violations */
    INT         *Ap, /* column pointers (size ncol+1) */
    INT         *Ai, /* row indices for A in increasing order for each
                        column (size Ap [ncol]) */
    double      *Ax, /* numerical entries of A (size Ap [ncol]) */
    double      *lo, /* lower bounds for x (size ncol) */
    double      *hi, /* upper bounds for x (size ncol) */
    double      *bl, /* lower bounds for b (size nrow) */
    double      *bu, /* upper bounds for b (size nrow) */
    double    *work  /* work array (size 2*nrow) */
) ;
