/*
 * copied from seti_boinc/validate/sah_result.h
 *
 * modified by Tetsuji Maverick Rai
 * further modified by Josef W. Segur
 *
 */
#ifndef _SAH_RESULT_
#define _SAH_RESULT_

// this file should not refer to either BOINC or S@h DB

#include <stdio.h>
#include <string.h> //uje
#include <vector>

extern double dQfin;
extern double dQtmp;
extern bool show_autocorrs;

using namespace std;

// Result Flags.  Can be passed from validator to assimilator
// via the BOINC result.opaque field.  Note that you have
// to cast to float in order to assign a flag to the opaque
// field.  OR into a temp int and then assign with a cast.
#define RESULT_FLAG_OVERFLOW 0x00000001


enum {
    SIGNAL_TYPE_SPIKE,
    SIGNAL_TYPE_AUTOCORR,
    SIGNAL_TYPE_GAUSSIAN,
    SIGNAL_TYPE_PULSE,
    SIGNAL_TYPE_TRIPLET,
    SIGNAL_TYPE_BEST_SPIKE,
    SIGNAL_TYPE_BEST_AUTOCORR,
    SIGNAL_TYPE_BEST_GAUSSIAN,
    SIGNAL_TYPE_BEST_PULSE,
    SIGNAL_TYPE_BEST_TRIPLET,
    N_SIGNAL_TYPES
};

struct SIGNAL {
    int type;
    double power;
    double period;
    double peak;
    double mean;
    double ra;
    double decl;
    double time;
    double freq;
    double sigma;
    double chisqr;
    double max_power;
    double peak_power;
    double mean_power;
    double score;
    unsigned char pot[256];
    int len_prof;
    double snr;
    double thresh;
    int fft_len;
    double chirp_rate;
    double delay;

    bool checked;   // temp

    bool roughly_equal(SIGNAL&, int);
};

struct SAH_RESULT {
    bool have_result;
    bool overflow;
    vector<SIGNAL> signals;

    bool has_roughly_equal_signal(SIGNAL&);
    bool has_nearly_equal_signal(SIGNAL&, int);
    bool strongly_similar(SAH_RESULT&);
    bool weakly_similar(SAH_RESULT&);
    bool bad_values();
    int parse_file(FILE*);
    int num_signals;
  void dump_signals();
};

extern int write_sah_db_entries(SAH_RESULT&);

#endif
