/* hmmsearch: search profile HMM(s) against a sequence database.
 *
OpenMP usage prototype
 */

/*
These are shortcuts to build on Cori. You're system will probably set something up different.
The only real difference is whatever flag your compiler uses to add openmp. -openmp -qopenmp
cc -openmp -O3 -ansi_alias  -fPIC  -DHAVE_CONFIG_H  -I../easel -I../libdivsufsort -I../easel -I. -I. -o hpc_hmmsearch.o -c hpc_hmmsearch.c
cc -openmp -O3 -ansi_alias  -fPIC  -DHAVE_CONFIG_H  -L../easel -L./impl_sse -L../libdivsufsort -L. -o hpc_hmmsearch hpc_hmmsearch.o  -lhmmer -leasel -ldivsufsort -lm
*/

#include "p7_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "easel.h"
#include "esl_alphabet.h"
#include "esl_getopts.h"
#include "esl_msa.h"
#include "esl_msafile.h"
#include "esl_sq.h"
#include "esl_sqio.h"

#include "hmmer.h"

//a global variable tracking how many work units remain.
//if it's less than the number of threads then we've become unbalanced 
//and we should consider subdividing work that remains
int work_counter;

typedef struct
{
  P7_BG *bg;
  P7_OPROFILE *om;
  P7_PIPELINE *pli;
  P7_TOPHITS *th;
} HMM_BUFFER;

typedef struct
{
  FILE *ofp;
  FILE *afp;
  FILE *tblfp;
  FILE *domtblfp;
  FILE *pfamtblfp;
  ESL_ALPHABET *abc;
  ESL_GETOPTS *go;
  int textw;
  int threads;
} OUTPUT_INFO;

//utility code has been moved to functions to make the openmp control flow more compact and readable
static int load_seq_buffer(ESL_SQFILE *dbfp, ESL_SQ **sbb, int seq_per_buffer);
static int load_hmm_buffer(P7_HMMFILE *hfp, HMM_BUFFER **hb, int *nquery, int buffer_size, ESL_ALPHABET *abc, ESL_GETOPTS *go);
static int output_hmm_buffer(HMM_BUFFER **hb, int buffer_size, int nquery, OUTPUT_INFO *oi);
static int thread_kernel(HMM_BUFFER *hb, ESL_SQ **sbb, int start, int end, ESL_GETOPTS *go, OUTPUT_INFO *oi);


#define REPOPTS     "-E,-T,--cut_ga,--cut_nc,--cut_tc"
#define DOMREPOPTS  "--domE,--domT,--cut_ga,--cut_nc,--cut_tc"
#define INCOPTS     "--incE,--incT,--cut_ga,--cut_nc,--cut_tc"
#define INCDOMOPTS  "--incdomE,--incdomT,--cut_ga,--cut_nc,--cut_tc"
#define THRESHOPTS  "-E,-T,--domE,--domT,--incE,--incT,--incdomE,--incdomT,--cut_ga,--cut_nc,--cut_tc"

static ESL_OPTIONS options[] = {
  /* name           type         default  env  range     toggles   reqs   incomp              help                                                      docgroup*/
  { "-h",           eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  NULL,            "show brief help on version and usage",                         1 },
  /* Control of output */
  { "-o",           eslARG_OUTFILE, NULL, NULL, NULL,    NULL,  NULL,  NULL,            "direct output to file <f>, not stdout",                        2 },
  { "-A",           eslARG_OUTFILE, NULL, NULL, NULL,    NULL,  NULL,  NULL,            "save multiple alignment of all hits to file <f>",              2 },
  { "--tblout",     eslARG_OUTFILE, NULL, NULL, NULL,    NULL,  NULL,  NULL,            "save parseable table of per-sequence hits to file <f>",        2 },
  { "--domtblout",  eslARG_OUTFILE, NULL, NULL, NULL,    NULL,  NULL,  NULL,            "save parseable table of per-domain hits to file <f>",          2 },
  { "--pfamtblout", eslARG_OUTFILE, NULL, NULL, NULL,    NULL,  NULL,  NULL,            "save table of hits and domains to file, in Pfam format <f>",   2 },
  { "--acc",        eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  NULL,            "prefer accessions over names in output",                       2 },
  { "--noali",      eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  NULL,            "don't output alignments, so output is smaller",                2 },
  { "--notextw",    eslARG_NONE,    NULL, NULL, NULL,    NULL,  NULL, "--textw",        "unlimit ASCII text output line width",                         2 },
  { "--textw",      eslARG_INT,    "120", NULL, "n>=120",NULL,  NULL, "--notextw",      "set max width of ASCII text output lines",                     2 },
  /* Control of reporting thresholds */
  { "-E",           eslARG_REAL,  "10.0", NULL, "x>0",   NULL,  NULL,  REPOPTS,         "report sequences <= this E-value threshold in output",         4 },
  { "-T",           eslARG_REAL,   FALSE, NULL, NULL,    NULL,  NULL,  REPOPTS,         "report sequences >= this score threshold in output",           4 },
  { "--domE",       eslARG_REAL,  "10.0", NULL, "x>0",   NULL,  NULL,  DOMREPOPTS,      "report domains <= this E-value threshold in output",           4 },
  { "--domT",       eslARG_REAL,   FALSE, NULL, NULL,    NULL,  NULL,  DOMREPOPTS,      "report domains >= this score cutoff in output",                4 },
  /* Control of inclusion (significance) thresholds */
  { "--incE",       eslARG_REAL,  "0.01", NULL, "x>0",   NULL,  NULL,  INCOPTS,         "consider sequences <= this E-value threshold as significant",  5 },
  { "--incT",       eslARG_REAL,   FALSE, NULL, NULL,    NULL,  NULL,  INCOPTS,         "consider sequences >= this score threshold as significant",    5 },
  { "--incdomE",    eslARG_REAL,  "0.01", NULL, "x>0",   NULL,  NULL,  INCDOMOPTS,      "consider domains <= this E-value threshold as significant",    5 },
  { "--incdomT",    eslARG_REAL,   FALSE, NULL, NULL,    NULL,  NULL,  INCDOMOPTS,      "consider domains >= this score threshold as significant",      5 },
  /* Model-specific thresholding for both reporting and inclusion */
  { "--cut_ga",     eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  THRESHOPTS,      "use profile's GA gathering cutoffs to set all thresholding",   6 },
  { "--cut_nc",     eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  THRESHOPTS,      "use profile's NC noise cutoffs to set all thresholding",       6 },
  { "--cut_tc",     eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL,  THRESHOPTS,      "use profile's TC trusted cutoffs to set all thresholding",     6 },
  /* Control of acceleration pipeline */
  { "--max",        eslARG_NONE,   FALSE, NULL, NULL,    NULL,  NULL, "--F1,--F2,--F3", "Turn all heuristic filters off (less speed, more power)",      7 },
  { "--F1",         eslARG_REAL,  "0.02", NULL, NULL,    NULL,  NULL, "--max",          "Stage 1 (MSV) threshold: promote hits w/ P <= F1",             7 },
  { "--F2",         eslARG_REAL,  "1e-3", NULL, NULL,    NULL,  NULL, "--max",          "Stage 2 (Vit) threshold: promote hits w/ P <= F2",             7 },
  { "--F3",         eslARG_REAL,  "1e-5", NULL, NULL,    NULL,  NULL, "--max",          "Stage 3 (Fwd) threshold: promote hits w/ P <= F3",             7 },
  { "--nobias",     eslARG_NONE,   NULL,  NULL, NULL,    NULL,  NULL, "--max",          "turn off composition bias filter",                             7 },

/* Other options */
  { "--nonull2",    eslARG_NONE,   NULL,  NULL, NULL,    NULL,  NULL,  NULL,            "turn off biased composition score corrections",               12 },
  { "-Z",           eslARG_REAL,   FALSE, NULL, "x>0",   NULL,  NULL,  NULL,            "set # of comparisons done, for E-value calculation",          12 },
  { "--domZ",       eslARG_REAL,   FALSE, NULL, "x>0",   NULL,  NULL,  NULL,            "set # of significant seqs, for domain E-value calculation",   12 },
  { "--seed",       eslARG_INT,    "42",  NULL, "n>=0",  NULL,  NULL,  NULL,            "set RNG seed to <n> (if 0: one-time arbitrary seed)",         12 },
  { "--tformat",    eslARG_STRING,  NULL, NULL, NULL,    NULL,  NULL,  NULL,            "assert target <seqfile> is in format <s>: no autodetection",  12 },

// thread buffer related parameters
  { "--seq_buffer", eslARG_INT, "200000", NULL, "n>=1", NULL, NULL, NULL,               "set # of sequences per thread buffer",                        13 },
  { "--hmm_buffer", eslARG_INT,     "500", NULL, "n>=1", NULL, NULL, NULL,               "set # of hmms per thread hmm buffer",                         13 },
  { "--cpu",        eslARG_INT,      "1", "OMP_NUM_THREADS", "n>=1", NULL, NULL, NULL,  "set # of threads",                                            13 },

  {  0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
};

static char usage[]  = "[options] <hmmfile> <seqdb>";
static char banner[] = "search profile(s) against a sequence database, custom modified for improved thread performance";

/* struct cfg_s : "Global" application configuration shared by all threads/processes
 * 
 * This structure is passed to routines within main.c, as a means of semi-encapsulation
 * of shared data amongst different parallel processes (threads or MPI processes).
The encapsulation here might be vestigal at this point since I needed to put alllll the
output details in a different object in order to compartmentalize results output from thread control.
 */
struct cfg_s 
{
  char            *dbfile;            /* target sequence database file                   */
  char            *hmmfile;           /* query HMM file                                  */

  char             *firstseq_key;     /* name of the first sequence in the restricted db range */
  int              n_targetseq;       /* number of sequences in the restricted range */
};

static int process_commandline(int argc, char **argv, ESL_GETOPTS **ret_go, char **ret_hmmfile, char **ret_seqfile)
{
  ESL_GETOPTS *go = esl_getopts_Create(options);
  int          status;

  if (esl_opt_ProcessEnvironment(go)         != eslOK)  { if (printf("Failed to process environment: %s\n", go->errbuf) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if (esl_opt_ProcessCmdline(go, argc, argv) != eslOK)  { if (printf("Failed to parse command line: %s\n",  go->errbuf) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if (esl_opt_VerifyConfig(go)               != eslOK)  { if (printf("Failed to parse command line: %s\n",  go->errbuf) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }

  /* help format: */
  if (esl_opt_GetBoolean(go, "-h") == TRUE) 
  {
    p7_banner(stdout, argv[0], banner);
    esl_usage(stdout, argv[0], usage);
    if (puts("\nBasic options:")                                           < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
    esl_opt_DisplayHelp(stdout, go, 1, 2, 80); /* 1= group; 2 = indentation; 80=textwidth*/

    if (puts("\nOptions directing output:")                                < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
    esl_opt_DisplayHelp(stdout, go, 2, 2, 80); 

    if (puts("\nOptions controlling reporting thresholds:")                < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
    esl_opt_DisplayHelp(stdout, go, 4, 2, 80); 

    if (puts("\nOptions controlling inclusion (significance) thresholds:") < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
    esl_opt_DisplayHelp(stdout, go, 5, 2, 80); 

    if (puts("\nOptions controlling model-specific thresholding:")         < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
    esl_opt_DisplayHelp(stdout, go, 6, 2, 80); 

    if (puts("\nOptions controlling acceleration heuristics:")             < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
    esl_opt_DisplayHelp(stdout, go, 7, 2, 80); 

    if (puts("\nExpert options:")                                          < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
    esl_opt_DisplayHelp(stdout, go, 12, 2, 80); 

    if (puts("\nInput buffer and thread control:")                         < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
    esl_opt_DisplayHelp(stdout, go, 13, 2, 80);
    exit(0);
  }

  if (esl_opt_ArgNumber(go)                  != 2)     { if (puts("Incorrect number of command line arguments.")      < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if ((*ret_hmmfile = esl_opt_GetArg(go, 1)) == NULL)  { if (puts("Failed to get <hmmfile> argument on command line") < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }
  if ((*ret_seqfile = esl_opt_GetArg(go, 2)) == NULL)  { if (puts("Failed to get <seqdb> argument on command line")   < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); goto FAILURE; }

  /* Validate any attempted use of stdin streams */
  if (strcmp(*ret_hmmfile, "-") == 0 && strcmp(*ret_seqfile, "-") == 0) 
  { 
    if (puts("Either <hmmfile> or <seqdb> may be '-' (to read from stdin), but not both.") < 0) 
      ESL_XEXCEPTION_SYS(eslEWRITE, "write failed"); 
    goto FAILURE; 
  }

  *ret_go = go;
  return eslOK;
  
FAILURE:  /* all errors handled here are user errors, so be polite.  */
  esl_usage(stdout, argv[0], usage);
  if (puts("\nwhere most common options are:")                                 < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
  esl_opt_DisplayHelp(stdout, go, 1, 2, 80); /* 1= group; 2 = indentation; 80=textwidth*/
  if (printf("\nTo see more help on available options, do %s -h\n\n", argv[0]) < 0) ESL_XEXCEPTION_SYS(eslEWRITE, "write failed");
  esl_getopts_Destroy(go);
  exit(1);  

ERROR:
  if (go) esl_getopts_Destroy(go);
  exit(status);
}

static int output_header(FILE *ofp, const ESL_GETOPTS *go, char *hmmfile, char *seqfile)
{
  p7_banner(ofp, go->argv[0], banner);
  
  if (fprintf(ofp, "# query HMM file:                  %s\n", hmmfile)                                                                                 < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (fprintf(ofp, "# target sequence database:        %s\n", seqfile)                                                                                 < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "-o")           && fprintf(ofp, "# output directed to file:         %s\n",             esl_opt_GetString(go, "-o"))           < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "-A")           && fprintf(ofp, "# MSA of all hits saved to file:   %s\n",             esl_opt_GetString(go, "-A"))           < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--tblout")     && fprintf(ofp, "# per-seq hits tabular output:     %s\n",             esl_opt_GetString(go, "--tblout"))     < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--domtblout")  && fprintf(ofp, "# per-dom hits tabular output:     %s\n",             esl_opt_GetString(go, "--domtblout"))  < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--pfamtblout") && fprintf(ofp, "# pfam-style tabular hit output:   %s\n",             esl_opt_GetString(go, "--pfamtblout")) < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--acc")        && fprintf(ofp, "# prefer accessions over names:    yes\n")                                                   < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--noali")      && fprintf(ofp, "# show alignments in output:       no\n")                                                    < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--notextw")    && fprintf(ofp, "# max ASCII text line length:      unlimited\n")                                             < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--textw")      && fprintf(ofp, "# max ASCII text line length:      %d\n",             esl_opt_GetInteger(go, "--textw"))     < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "-E")           && fprintf(ofp, "# sequence reporting threshold:    E-value <= %g\n",  esl_opt_GetReal(go, "-E"))             < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "-T")           && fprintf(ofp, "# sequence reporting threshold:    score >= %g\n",    esl_opt_GetReal(go, "-T"))             < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--domE")       && fprintf(ofp, "# domain reporting threshold:      E-value <= %g\n",  esl_opt_GetReal(go, "--domE"))         < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--domT")       && fprintf(ofp, "# domain reporting threshold:      score >= %g\n",    esl_opt_GetReal(go, "--domT"))         < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--incE")       && fprintf(ofp, "# sequence inclusion threshold:    E-value <= %g\n",  esl_opt_GetReal(go, "--incE"))         < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--incT")       && fprintf(ofp, "# sequence inclusion threshold:    score >= %g\n",    esl_opt_GetReal(go, "--incT"))         < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--incdomE")    && fprintf(ofp, "# domain inclusion threshold:      E-value <= %g\n",  esl_opt_GetReal(go, "--incdomE"))      < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--incdomT")    && fprintf(ofp, "# domain inclusion threshold:      score >= %g\n",    esl_opt_GetReal(go, "--incdomT"))      < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--cut_ga")     && fprintf(ofp, "# model-specific thresholding:     GA cutoffs\n")                                            < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); 
  if (esl_opt_IsUsed(go, "--cut_nc")     && fprintf(ofp, "# model-specific thresholding:     NC cutoffs\n")                                            < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); 
  if (esl_opt_IsUsed(go, "--cut_tc")     && fprintf(ofp, "# model-specific thresholding:     TC cutoffs\n")                                            < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--max")        && fprintf(ofp, "# Max sensitivity mode:            on [all heuristic filters off]\n")                        < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--F1")         && fprintf(ofp, "# MSV filter P threshold:       <= %g\n",             esl_opt_GetReal(go, "--F1"))           < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--F2")         && fprintf(ofp, "# Vit filter P threshold:       <= %g\n",             esl_opt_GetReal(go, "--F2"))           < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--F3")         && fprintf(ofp, "# Fwd filter P threshold:       <= %g\n",             esl_opt_GetReal(go, "--F3"))           < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--nobias")     && fprintf(ofp, "# biased composition HMM filter:   off\n")                                                   < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");

  if (esl_opt_IsUsed(go, "--seq_buffer") && fprintf(ofp, "# sequences per sequence buffer:       <= %d\n",    esl_opt_GetInteger(go, "--seq_buffer"))       < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--hmm_buffer")       && fprintf(ofp, "# hmms per hmm buffer       <= %d\n",                     esl_opt_GetInteger(go, "--hmm_buffer"))             < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--cpu")    && fprintf(ofp, "# threads                   <= %d\n",                     esl_opt_GetInteger(go, "--cpu"))             < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");

  if (esl_opt_IsUsed(go, "--nonull2")    && fprintf(ofp, "# null2 bias corrections:          off\n")                                                   < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "-Z")           && fprintf(ofp, "# sequence search space set to:    %.0f\n",           esl_opt_GetReal(go, "-Z"))             < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--domZ")       && fprintf(ofp, "# domain search space set to:      %.0f\n",           esl_opt_GetReal(go, "--domZ"))         < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (esl_opt_IsUsed(go, "--seed"))  
  {
    if (esl_opt_GetInteger(go, "--seed") == 0 && fprintf(ofp, "# random number seed:              one-time arbitrary\n")                               < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
    else if (                               fprintf(ofp, "# random number seed set to:       %d\n",             esl_opt_GetInteger(go, "--seed"))      < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  }
  if (esl_opt_IsUsed(go, "--tformat")    && fprintf(ofp, "# targ <seqfile> format asserted:  %s\n",             esl_opt_GetString(go, "--tformat"))    < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  if (fprintf(ofp, "# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n\n")                                                    < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed");
  return eslOK;
}

int main(int argc, char **argv)
{
  ESL_GETOPTS     *go       = NULL;	
  struct cfg_s     cfg;        
  int              status   = eslOK;
  OUTPUT_INFO      oi;

  impl_Init();                  /* processor specific initialization */
  p7_FLogsumInit();		/* we're going to use table-driven Logsum() approximations at times */

  /* Initialize what we can in the config structure (without knowing the alphabet yet) 
   */
  cfg.hmmfile    = NULL;
  cfg.dbfile     = NULL;
  cfg.firstseq_key = NULL;
  cfg.n_targetseq  = -1;

  process_commandline(argc, argv, &go, &cfg.hmmfile, &cfg.dbfile);    

/* is the range restricted? */

  oi.ofp = stdout;
  oi.afp = NULL;
  oi.tblfp = NULL;
  oi.domtblfp = NULL;
  oi.pfamtblfp = NULL;
  oi.abc = NULL;
  oi.go = go;
  
  P7_HMMFILE *hfp = NULL;
  ESL_SQFILE *dbfp = NULL;
  P7_HMM *hmm = NULL;
  int dbfmt = eslSQFILE_UNKNOWN;

  int nquery = 0;

  int hstatus;
  int sstatus;

  char             errbuf[eslERRBUFSIZE];

  if (esl_opt_GetBoolean(go, "--notextw")) oi.textw = 0;
  else                                     oi.textw = esl_opt_GetInteger(go, "--textw");

  if (esl_opt_IsOn(go, "--tformat")) 
  {
    dbfmt = esl_sqio_EncodeFormat(esl_opt_GetString(go, "--tformat"));
    if (dbfmt == eslSQFILE_UNKNOWN) p7_Fail("%s is not a recognized sequence database file format\n", esl_opt_GetString(go, "--tformat"));
  }

  /* Open the target sequence database */
  status = esl_sqfile_Open(cfg.dbfile, dbfmt, p7_SEQDBENV, &dbfp);
  if      (status == eslENOTFOUND) p7_Fail("Failed to open sequence file %s for reading\n",          cfg.dbfile);
  else if (status == eslEFORMAT)   p7_Fail("Sequence file %s is empty or misformatted\n",            cfg.dbfile);
  else if (status == eslEINVAL)    p7_Fail("Can't autodetect format of a stdin or .gz seqfile");
  else if (status != eslOK)        p7_Fail("Unexpected error %d opening sequence file %s\n", status, cfg.dbfile);  

  //move this forward a bit so that the output_header has correct output file handle
  if (esl_opt_IsOn(go, "-o"))          { if ((oi.ofp      = fopen(esl_opt_GetString(go, "-o"), "w")) == NULL) p7_Fail("Failed to open output file %s for writing\n",    esl_opt_GetString(go, "-o")); }

  /* Open the query profile HMM file */
  status = p7_hmmfile_OpenE(cfg.hmmfile, NULL, &hfp, errbuf);
  if      (status == eslENOTFOUND) p7_Fail("File existence/permissions problem in trying to open HMM file %s.\n%s\n", cfg.hmmfile, errbuf);
  else if (status == eslEFORMAT)   p7_Fail("File format problem in trying to open HMM file %s.\n%s\n",                cfg.hmmfile, errbuf);
  else if (status != eslOK)        p7_Fail("Unexpected error %d in opening HMM file %s.\n%s\n",               status, cfg.hmmfile, errbuf);
  /* <abc> is not known 'til first HMM is read. */
  hstatus = p7_hmmfile_Read(hfp, &oi.abc, &hmm);
  if (hstatus == eslOK)
  {
    /* One-time initializations after alphabet <abc> becomes known */
    output_header(oi.ofp, go, cfg.hmmfile, cfg.dbfile);
    esl_sqfile_SetDigital(dbfp, oi.abc); //ReadBlock requires knowledge of the alphabet to decide how best to read blocks
  }
  p7_hmmfile_Close(hfp);
  status = p7_hmmfile_OpenE(cfg.hmmfile, NULL, &hfp, errbuf);
  if(status != eslOK) p7_Fail("Reopening the hmm shouldn't have failed.\n");

  /* Open the results output files */
  if (esl_opt_IsOn(go, "-A"))          { if ((oi.afp      = fopen(esl_opt_GetString(go, "-A"), "w")) == NULL) p7_Fail("Failed to open alignment file %s for writing\n", esl_opt_GetString(go, "-A")); }
  if (esl_opt_IsOn(go, "--tblout"))    { if ((oi.tblfp    = fopen(esl_opt_GetString(go, "--tblout"),    "w")) == NULL)  esl_fatal("Failed to open tabular per-seq output file %s for writing\n", esl_opt_GetString(go, "--tblout")); }
  if (esl_opt_IsOn(go, "--domtblout")) { if ((oi.domtblfp = fopen(esl_opt_GetString(go, "--domtblout"), "w")) == NULL)  esl_fatal("Failed to open tabular per-dom output file %s for writing\n", esl_opt_GetString(go, "--domtblout")); }
  if (esl_opt_IsOn(go, "--pfamtblout")){ if ((oi.pfamtblfp = fopen(esl_opt_GetString(go, "--pfamtblout"), "w")) == NULL)  esl_fatal("Failed to open pfam-style tabular output file %s for writing\n", esl_opt_GetString(go, "--pfamtblout")); }
  

int seq_buffer_size = esl_opt_GetInteger(go, "--seq_buffer");
int hmm_buffer_size = esl_opt_GetInteger(go, "--hmm_buffer");
int requested_threads = esl_opt_GetInteger(go, "--cpu");

oi.threads = requested_threads;

//now build some data structures to contain the data buffers

// prepare the sequence block buffer
// The aim is to alternate between flip and flop. one buffer is being read from file
// while the second is being processed.

  ESL_SQ **sbb_flip     = NULL, **sbb_flop     = NULL;

  ESL_ALLOC(sbb_flip     , sizeof(ESL_SQ*) * seq_buffer_size);
  ESL_ALLOC(sbb_flop     , sizeof(ESL_SQ*) * seq_buffer_size);

  int a;
  for(a = 0; a < seq_buffer_size; a++)
  {
    sbb_flip[a] = esl_sq_CreateDigital(oi.abc);
    sbb_flop[a] = esl_sq_CreateDigital(oi.abc); 
  }

 //build the hmm buffer blocks
  HMM_BUFFER **hb_flip     = NULL, **hb_flop     = NULL;
  HMM_BUFFER  *hb_flip_mem = NULL,  *hb_flop_mem = NULL;

  ESL_ALLOC(hb_flip    , sizeof(HMM_BUFFER*) * hmm_buffer_size);
  ESL_ALLOC(hb_flop    , sizeof(HMM_BUFFER*) * hmm_buffer_size);
  ESL_ALLOC(hb_flip_mem, sizeof(HMM_BUFFER ) * hmm_buffer_size);
  ESL_ALLOC(hb_flop_mem, sizeof(HMM_BUFFER ) * hmm_buffer_size);

  for(a = 0; a < hmm_buffer_size; a++)
  {
    hb_flip[a] = hb_flip_mem + a;
    hb_flop[a] = hb_flop_mem + a;

    hb_flip[a]->pli = NULL; 
    hb_flip[a]->th  = NULL;
    hb_flip[a]->om  = NULL;
    hb_flip[a]->bg  = NULL;  
    hb_flop[a]->pli = NULL;
    hb_flop[a]->th  = NULL;
    hb_flop[a]->om  = NULL;
    hb_flop[a]->bg  = NULL;
  }

  //first step: prime the pipeline by reading the first seq buffer and the first hmm buffer into flip
  #pragma omp parallel num_threads(requested_threads)
  {
    //this single thread serializes high level control flow decisions such as when to flip buffers
    //and when to load/process buffers. I tried a few other arrangements (involving shared loop control variables) but kept getting
    //threads leaking through loop conditionals, stuck on the wrong barriers, and hanging the program
    #pragma omp single
    {
      //load first seq buffer/ta
      #pragma omp task
      { sstatus = load_seq_buffer(dbfp, sbb_flip, seq_buffer_size); }
      //load the first hmm buffer
      #pragma omp task
      { hstatus = load_hmm_buffer(hfp, hb_flip, &nquery, hmm_buffer_size, oi.abc, go); }

      // do not proceed until the flip buffers have their first inputs ready
      #pragma omp taskwait 

      //special case when the entire seq db fits in one buffer. Skip reading any more from seq file and skip flipping seq buffers.
      //remember we're in the single control thread scope so this is not a shared control variable
      int stabilize_seq = 0;
      if(sstatus == eslEOF) stabilize_seq = 1;

      while(hstatus == eslOK) //while we have not yet encountered a partially full model buffer, iterate the pipeline thorugh another model buffer
      {
        //two steps to prepare the next hmm buffer: output results currently in flop (unless this is the first iteration), then
        //read the next portion of the hmm file into flop
        #pragma omp task 
        {
          if(hb_flop[0]->om != NULL) //this bails if this is the first iteration before anything has been processed
          { output_hmm_buffer(hb_flop, hmm_buffer_size, nquery, &oi); }

          hstatus = load_hmm_buffer(hfp, hb_flop, &nquery, hmm_buffer_size, oi.abc, go);
        }

        do //this is the sequence buffer loop. do at least once (where the initial load was a partial and sstatus is now eslEOF)
        {
          //this taskgroup encloses an entire work block including child tasks spawned by load balancing inside the work kernel
          #pragma omp taskgroup
          {
            //flip contains entire seqdb, don't bother with seq flop
            if(stabilize_seq == 0);
            {
              #pragma omp task 
              { sstatus = load_seq_buffer(dbfp, sbb_flop, seq_buffer_size); }
            }

            work_counter = 0;
            //now create the tasks for the currenty body of work which is every hmm in flip buffer X every block in seq flip buffer
            int hb_idx;
            for(hb_idx = 0; hb_idx < hmm_buffer_size; hb_idx++)
            {  
              #pragma omp atomic
              work_counter++;
              #pragma omp task 
              { 
                thread_kernel(hb_flip[hb_idx], sbb_flip, 0, seq_buffer_size, go, &oi); 
              }
            }
          }
          //task group acts as barrier on the preparation of the next seq buffer and the completion of all work units

          if(stabilize_seq == 0)
          {
            ESL_SQ **temp = sbb_flip;
            sbb_flip = sbb_flop;
            sbb_flop = temp;
          } 
        } while(sstatus != eslEOF);

        //now do a final iteration of the seq buffer loop. this will load the first block of the next pass through the seq buffer
        //and process the last seq buffer of the current model buffer

        if(stabilize_seq == 0) //if one seq buffer holds all sequences, then we're already done
        {
          //work block task group
          #pragma omp taskgroup
          {
            #pragma omp task 
            { sstatus = load_seq_buffer(dbfp, sbb_flop, seq_buffer_size); }

            int hb_idx;
            work_counter = 0;
            for(hb_idx = 0; hb_idx < hmm_buffer_size; hb_idx++)
            {
              #pragma omp atomic
              work_counter++;
              #pragma omp task 
              { 
                thread_kernel(hb_flip[hb_idx], sbb_flip, 0, seq_buffer_size, go, &oi); 
              }
            }
          } //task group barrier to complete work block

          if(stabilize_seq == 0)
          {
            ESL_SQ **temp = sbb_flip;
            sbb_flip = sbb_flop;
            sbb_flop = temp;
          }
        }

        //this taskwait catches the unlikely situation where all processing is completed before
        //the hmm buffer loading task is finished
        #pragma omp taskwait 
 
        HMM_BUFFER **temp = hb_flip;
        hb_flip = hb_flop;
        hb_flop = temp;
      } 
      //back down to the scope of the single control thread

      //output of the last full buffer of models 
      #pragma omp task
      {
        output_hmm_buffer(hb_flop, hmm_buffer_size, nquery, &oi);
      }

      //if hmm_buffer_size evenly divides the number of models, then the final "partial" buffer is actually empty
      //and this setion can be completely skipped
      if(hb_flip[0]->om != NULL)
      {
        //process the final partially full hmm buffer in flip
        do
        {
          //work block task group
          #pragma omp taskgroup
          {
            #pragma omp task 
            {
              if(stabilize_seq == 0)
                sstatus = load_seq_buffer(dbfp, sbb_flop, seq_buffer_size);
            }

            int hb_idx;
            work_counter = 0;
            for(hb_idx = 0; hb_idx < hmm_buffer_size; hb_idx++)
            {
              #pragma omp atomic
              work_counter++;
              #pragma omp task 
              {
                thread_kernel(hb_flip[hb_idx], sbb_flip, 0, seq_buffer_size, go, &oi); 
              }
            }
          }

          if(stabilize_seq == 0)
          {
            ESL_SQ **temp = sbb_flip;
            sbb_flip = sbb_flop;
            sbb_flop = temp;
          } 
        } while(sstatus != eslEOF);

        if(stabilize_seq == 0)
        {
          //work block task group
          #pragma omp taskgroup
          {
            int hb_idx;
            work_counter = 0;
            for(hb_idx = 0; hb_idx < hmm_buffer_size; hb_idx++)
            {
              #pragma omp atomic
              work_counter++;
              #pragma omp task
              { 
                thread_kernel(hb_flip[hb_idx], sbb_flip, 0, seq_buffer_size, go, &oi); 
              }
            }
          } //final work block task group barrier
        }
      } 
    } //end single control thread
  } //end parallel region

  //finally, write the output for the work on the partial hmm buffer
  output_hmm_buffer(hb_flip, hmm_buffer_size, nquery, &oi);

  /* Terminate outputs... any last words?
   */
  if (oi.tblfp)    p7_tophits_TabularTail(oi.tblfp,    "hmmsearch", p7_SEARCH_SEQS, cfg.hmmfile, cfg.dbfile, go);
  if (oi.domtblfp) p7_tophits_TabularTail(oi.domtblfp, "hmmsearch", p7_SEARCH_SEQS, cfg.hmmfile, cfg.dbfile, go);
  if (oi.pfamtblfp) p7_tophits_TabularTail(oi.pfamtblfp,"hmmsearch", p7_SEARCH_SEQS, cfg.hmmfile, cfg.dbfile, go);
  if (oi.ofp)      { if (fprintf(oi.ofp, "[ok]\n") < 0) ESL_EXCEPTION_SYS(eslEWRITE, "write failed"); }
  esl_getopts_Destroy(go);

  free(hb_flip); free(hb_flip_mem);
  free(hb_flop); free(hb_flop_mem);

  for(a = 0; a < seq_buffer_size; a++)
  {
    esl_sq_Destroy(sbb_flip[a]);
    esl_sq_Destroy(sbb_flop[a]);
  }
  free(sbb_flip);
  free(sbb_flop);

  if (oi.ofp != stdout) fclose(oi.ofp);
  if (oi.afp)           fclose(oi.afp);
  if (oi.tblfp)         fclose(oi.tblfp);
  if (oi.domtblfp)      fclose(oi.domtblfp);
  if (oi.pfamtblfp)     fclose(oi.pfamtblfp);

  return eslOK;

ERROR:
  printf("oh geez a heap problem of some sort\n");
  return eslOK;
}

//load a number of sequences from the file into the given sequence buffer
//if the end of file is hit, the remaining block space is 0 length sequences (what esl_sq_Reuse makes)
//return eslOK if the entire seq buffer holds new data and there could be more in the file
//if the seq buffer was not filled because EOF, then reset its position to the
//start and return eslEOF
static int load_seq_buffer(ESL_SQFILE *dbfp, ESL_SQ **sbb, int seq_per_buffer)
{
  int x, y;
  int sstatus = eslOK;
  static int count = 0;

  for(x = 0; x < seq_per_buffer; x++)
  {
    esl_sq_Reuse(sbb[x]);
    if((sstatus = esl_sqio_Read(dbfp, sbb[x])) == eslOK)
      count++;
  }

  switch(sstatus)
  {
    case eslEFORMAT: fprintf(stderr, "Parse failed (sequence file %s):\n%s\n", dbfp->filename, esl_sqfile_GetErrorBuf(dbfp)); exit(0); break;
    case eslEOF    :
      count = 0;
      int s = esl_sqfile_Position(dbfp, 0);
      if(s != eslOK)
        p7_Fail("Failure rewinding sequence file\n");
      break;
    case eslOK    : /* do nothing */ break;
    default       : fprintf(stderr, "Unexpected error %d reading sequence file %s", sstatus, dbfp->filename); exit(0);
  }

  return sstatus;
}

//load from a hmm file into the given hmm buffer
//return eslOK if the hmm buffer was filled completely. 
//partial buffers are padded with NULL oprofiles
//or whatever state they are in after being Reused()
//return eslEOF if the buffer was partially filled with new data
static int load_hmm_buffer(P7_HMMFILE *hfp, HMM_BUFFER **hb, int *nquery, int buffer_size, ESL_ALPHABET *abc, ESL_GETOPTS *go)
{
  int x;
  int hstatus = eslOK;

  P7_HMM *hmm = NULL;
  P7_PROFILE *gm = NULL;
  
  for(x = 0; x < buffer_size; x++)
  {
    hstatus = p7_hmmfile_Read(hfp, &abc, &hmm);

    switch (hstatus)
    {
      case eslEOD      : p7_Fail("read failed, HMM file may be truncated?"); break;
      case eslEFORMAT  : p7_Fail("bad file format in HMM file "           ); break;
      case eslEINCOMPAT: p7_Fail("HMM file contains different alphabets"  ); break;
      case eslEOF      :                                                     break;
      case eslOK       :       
        *nquery++;
        gm = p7_profile_Create(hmm->M, abc);
        hb[x]->om = p7_oprofile_Create(hmm->M, abc);
        hb[x]->bg = p7_bg_Create(abc);
        p7_ProfileConfig(hmm, hb[x]->bg, gm, 10, p7_LOCAL);
        p7_oprofile_Convert(gm, hb[x]->om);
        //this pipeline never actually gets used, it's just merged statistics counts from
        //all the worker copies. it needs no dp working memory
        hb[x]->pli = p7_pipeline_Create(go, 1, 1, FALSE, p7_SEARCH_SEQS);

        hb[x]->th = p7_tophits_Create();
        p7_pli_NewModel(hb[x]->pli, hb[x]->om, hb[x]->bg);

        p7_hmm_Destroy(hmm);
        p7_profile_Destroy(gm);
        break;
      default:
        p7_Fail("Unexpected error (%d) in reading HMMs", hstatus);
    }
  }

  return hstatus;
}

//go into the hmm buffer and output its contents
//stop outputting when null model data is found (meaning it's a partial or empty block)
static int output_hmm_buffer(HMM_BUFFER **hb, int buffer_size, int nquery, OUTPUT_INFO *oi)
{
  int x;

  FILE       *ofp = oi->ofp;
  FILE       *afp = oi->afp;
  FILE     *tblfp = oi->tblfp;
  FILE  *domtblfp = oi->domtblfp;
  FILE *pfamtblfp = oi->pfamtblfp;

  ESL_ALPHABET *abc = oi->abc;
  ESL_GETOPTS   *go = oi->go;

  for(x = 0; x < buffer_size; x++)
  {
    if(hb[x]->om == NULL)
      break;
    else 
    {
      if (fprintf(ofp, "Query:       %s  [M=%d]\n", hb[x]->om->name, hb[x]->om->M)  < 0) { fprintf(stderr, "output write failed\n"); exit(0); }
      if (hb[x]->om->acc) { if (fprintf(ofp, "Accession:   %s\n", hb[x]->om->acc)   < 0) { fprintf(stderr, "output write failed\n"); exit(0); } } 
      if (hb[x]->om->desc) { if (fprintf(ofp, "Description: %s\n", hb[x]->om->desc) < 0) { fprintf(stderr, "output write failed\n"); exit(0); } }

      p7_tophits_SortBySortkey(hb[x]->th);
      p7_tophits_Threshold(hb[x]->th, hb[x]->pli);
      p7_tophits_Targets(ofp, hb[x]->th, hb[x]->pli, oi->textw);
      if (fprintf(ofp, "\n\n") < 0) { fprintf(stderr, "output write failed\n"); exit(0); }

      p7_tophits_Domains(ofp, hb[x]->th, hb[x]->pli, oi->textw);
      if (fprintf(ofp, "\n\n") < 0) { fprintf(stderr, "output write failed\n"); exit(0); }
 
      if (tblfp)     p7_tophits_TabularTargets (    tblfp, hb[x]->om->name, hb[x]->om->acc, hb[x]->th, hb[x]->pli, (nquery == 1));
      if (domtblfp)  p7_tophits_TabularDomains ( domtblfp, hb[x]->om->name, hb[x]->om->acc, hb[x]->th, hb[x]->pli, (nquery == 1));
      if (pfamtblfp) p7_tophits_TabularXfam    (pfamtblfp, hb[x]->om->name, hb[x]->om->acc, hb[x]->th, hb[x]->pli               );
  
      p7_pli_Statistics(ofp, hb[x]->pli, NULL);
      if (fprintf(ofp, "//\n") < 0) { fprintf(stderr, "output write failed\n"); exit(0); }

      /* Output the results in an MSA (-A option) */
      if (afp)
      {
        ESL_MSA *msa = NULL;

        if (p7_tophits_Alignment(hb[x]->th, abc, NULL, NULL, 0, p7_ALL_CONSENSUS_COLS, &msa) == eslOK)
        {
          if (oi->textw > 0) esl_msafile_Write(afp, msa, eslMSAFILE_STOCKHOLM);
          else               esl_msafile_Write(afp, msa, eslMSAFILE_PFAM);
          if (fprintf(ofp, "# Alignment of %d hits satisfying inclusion thresholds saved to: %s\n", msa->nseq, esl_opt_GetString(go, "-A")) < 0)
            { fprintf(stderr, "output write failed\n"); exit(0); }
        }
        else
          if (fprintf(ofp, "# No hits satisfy inclusion thresholds; no alignment saved\n") < 0) { fprintf(stderr, "output write failed\n"); exit(1); }

        esl_msa_Destroy(msa);
      }

      p7_pipeline_Destroy(hb[x]->pli);
      p7_tophits_Destroy (hb[x]->th );
      p7_oprofile_Destroy(hb[x]->om );
      p7_bg_Destroy      (hb[x]->bg );

      hb[x]->om = NULL;
    }
  }

  return eslOK;
}

static int thread_kernel(HMM_BUFFER *hb, ESL_SQ **sbb, int start, int end, ESL_GETOPTS *go, OUTPUT_INFO *oi)
{
  if(hb->om != NULL && sbb[0]->n > 0) //if either the model or the seq is empty then just skip it
  {
    //make working copies of all needed data structures for a work unit
    P7_OPROFILE *om  = p7_oprofile_Copy(hb->om);
    P7_TOPHITS  *th  = p7_tophits_Create();
    P7_BG       *bg  = p7_bg_Create(oi->abc);
    P7_PIPELINE *pli = p7_pipeline_Create(go, om->M, 100, FALSE, p7_SEARCH_SEQS);
                       p7_pli_NewModel(pli, om, bg);

    int x; 
    for(x = start; x < end; x++)
    {
      //when fewer tasks remain than available threads then we are losing time.
      //we can take a remaining task and divert some of its sequences into a new task.
      //this 8 is arbitrary right now. future work to measure new task overhead and set accordingly
      if((work_counter <= (oi->threads)) && (x < (end - 8)))
      {
        #pragma omp atomic
        work_counter++;

        int tx = x;
        x = x + ((end - x) >> 1);
        
        #pragma omp task
        {
          thread_kernel(hb, sbb, tx, x, go, oi);
        }
      }


      if(sbb[x]->n > 0)
      {
        p7_pli_NewSeq(pli, sbb[x]);
        p7_bg_SetLength(bg, sbb[x]->n);
        p7_oprofile_ReconfigLength(om, sbb[x]->n);
        p7_Pipeline(pli, om, bg, sbb[x], NULL, th);
        p7_pipeline_Reuse(pli);
      }
    }

    //take the results of this work unit and merge them with the master results in the hmm buffer
    #pragma omp critical
    {
      p7_tophits_Merge(hb->th, th);
      p7_pipeline_Merge(hb->pli, pli);
    }

    p7_oprofile_Destroy(om);
    p7_tophits_Destroy(th);
    p7_pipeline_Destroy(pli);
    p7_bg_Destroy(bg);
  }

  #pragma omp atomic 
  work_counter--;

  return eslOK;
}

