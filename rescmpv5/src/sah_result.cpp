/*
 * copied from seti_boinc/validate/sah_result.cpp
 *
 * modified by Tetsuji Maverick Rai
 * further modified by Josef W. Segur
 *
 */
#include <cstdio>
#include <cmath>
#include <iostream>
#include <iomanip>


#include "parse.h"
#include "sah_result.h"

using namespace std;

// the difference between two numbers,
// as a fraction of the largest in absolute value
//
double rel_diff(double x1, double x2) {
    if (x1 == 0 && x2 == 0) return 0;
    if (x1 == 0) return 1;
    if (x2 == 0) return 1;

    double d1 = fabs(x1);
    double d2 = fabs(x2);
    double d = fabs(x1-x2);
    if (d1 > d2) return d/d1;
    return d/d2;
}

double abs_diff(double x1, double x2) {
    return fabs(x1-x2);
}

// return true if the two signals are the same
// within numerical tolerances
//
// JWS: Starting with Rescmpv4 almost all important parameters are included in
// the comparison, derived from the modified validation code for SET@home v7.
// As with earlier versions, for signals which pass the "good enough" test,
// the code keeps track of deviations from exact equality so that even slight
// inaccuracies can be reported as a "Q" value.

bool SIGNAL::roughly_equal(SIGNAL& s, int tight) {
    double second = 1.0/86400.0;	// 1 second as a fraction of a day
    double dQt;

    if (type != s.type) return false;

    // tolerances common to all signals
    if ((dQt=abs_diff(ra, s.ra))                 > .00066) return false; // .01 deg
    if (tight == 3 && dQt                           > 0.0) return false; // exact
    dQtmp = dQt*15;

    if ((dQt = abs_diff(decl, s.decl))              > .01) return false; // .01 deg
    if (tight == 3 && dQt                           > 0.0) return false; // exact
    if (dQt > dQtmp) dQtmp = dQt;

    if ((dQt=abs_diff(time, s.time))             > second) return false; // 1 sec
    if (tight == 3 && dQt                           > 0.0) return false; // exact
    dQt *= 864;
    if (dQt > dQtmp) dQtmp = dQt;

    if ((dQt=abs_diff(freq, s.freq))                > .01) return false; // .01 Hz
    if (tight == 1 && dQt                          > .001) return false; // .001 Hz
    if (tight == 2 && dQt                         > .0001) return false; // .0001 Hz
    if (tight == 3 && dQt                           > 0.0) return false; // exact
    if (dQt > dQtmp) dQtmp = dQt;

    if ((dQt=abs_diff(chirp_rate, s.chirp_rate))    > .01) return false; // .01 Hz/s
    if (tight == 1 && dQt                          > .001) return false; // .001 Hz/s
    if (tight == 2 && dQt                         > .0001) return false; // .0001 Hz/s
    if (tight == 3 && dQt                           > 0.0) return false; // exact
    if (dQt > dQtmp) dQtmp = dQt;

    if (fft_len != s.fft_len) 						       return false; // equal

    // peak_power is actually common, but not treated so in the official validator.
    // A large difference will give Q=???? to show something bad for the loose check.
    dQt=rel_diff(peak_power, s.peak_power);
    if (tight == 1 && dQt                          > .001) return false; // 0.1 %
    if (tight == 2 && dQt                         > .0001) return false; // 0.01 %
    if (tight == 3 && dQt                           > 0.0) return false; // exact
    if (dQt > dQtmp) dQtmp = dQt;

    switch(type) {
        case SIGNAL_TYPE_SPIKE:
        case SIGNAL_TYPE_BEST_SPIKE:
            // Pretend there's a "power"
            if ((dQt=rel_diff(power, s.power))      > .01) return false; // 1%
            if (dQt > dQtmp) dQtmp = dQt;
            return true;
        case SIGNAL_TYPE_GAUSSIAN:
        case SIGNAL_TYPE_BEST_GAUSSIAN:
            if (rel_diff(peak_power, s.peak_power)  > .01) return false; // 1%

            if ((dQt=rel_diff(mean_power, s.mean_power))  > .01) return false; // 1%
            if (tight == 1 && dQt                  > .001) return false; // 0.1 %
            if (tight == 2 && dQt                 > .0001) return false; // 0.01 %
            if (tight == 3 && dQt                   > 0.0) return false; // exact
            if (dQt > dQtmp) dQtmp = dQt;

            if ((dQt=rel_diff(sigma, s.sigma))      > .01) return false; // 1%
            if (dQt > dQtmp) dQtmp = dQt;

            if ((dQt=rel_diff(chisqr, s.chisqr))    > .01) return false; // 1%
            if (tight == 1 && dQt                  > .001) return false; // 0.1 %
            if (tight == 2 && dQt                 > .0001) return false; // 0.01 %
            if (tight == 3 && dQt                   > 0.0) return false; // exact
            if (dQt > dQtmp) dQtmp = dQt;
            return true;
        case SIGNAL_TYPE_PULSE:
        case SIGNAL_TYPE_BEST_PULSE:
            // Pretend there's a "power"
            if ((dQt=rel_diff(power, s.power))      > .01) return false; // 1%
            if (dQt > dQtmp) dQtmp = dQt;

            if ((dQt=rel_diff(mean_power, s.mean_power))  > .01) return false; // 1%
            if (tight == 1 && dQt                  > .001) return false; // 0.1 %
            if (tight == 2 && dQt                 > .0001) return false; // 0.01 %
            if (tight == 3 && dQt                   > 0.0) return false; // exact
            if (dQt > dQtmp) dQtmp = dQt;

            if ((dQt=abs_diff(period, s.period))    > .01) return false; // 1%
            if (tight == 1 && dQt                  > .001) return false; // 0.1 %
            if (tight == 2 && dQt                 > .0001) return false; // 0.01 %
            if (tight == 3 && dQt                   > 0.0) return false; // exact
            if (dQt > dQtmp) dQtmp = dQt;

            if ((dQt=rel_diff(snr, s.snr))          > .01) return false; // 1%
            if (tight == 1 && dQt                  > .001) return false; // 0.1 %
            if (tight == 2 && dQt                 > .0001) return false; // 0.01 %
            if (tight == 3 && dQt                   > 0.0) return false; // exact
            if (dQt > dQtmp) dQtmp = dQt;

            if ((dQt=rel_diff(thresh, s.thresh))    > .01) return false; // 1%
            if (tight == 1 && dQt                  > .001) return false; // 0.1 %
            if (tight == 2 && dQt                 > .0001) return false; // 0.01 %
            if (tight == 3 && dQt                   > 0.0) return false; // exact
            if (dQt > dQtmp) dQtmp = dQt;
            return true;
        case SIGNAL_TYPE_TRIPLET:
        case SIGNAL_TYPE_BEST_TRIPLET:
            // Pretend there's a "power"
            if ((dQt=rel_diff(power, s.power))      > .01) return false; // 1%
            if (dQt > dQtmp) dQtmp = dQt;

            if ((dQt=rel_diff(period, s.period))    > .01) return false; // 1%
            if (tight == 1 && dQt                  > .001) return false; // 0.1 %
            if (tight == 2 && dQt                 > .0001) return false; // 0.01 %
            if (tight == 3 && dQt                   > 0.0) return false; // exact
            if (dQt > dQtmp) dQtmp = dQt;
            return true;
        case SIGNAL_TYPE_AUTOCORR:
        case SIGNAL_TYPE_BEST_AUTOCORR:
            // Pretend there's a "power"
            if ((dQt=rel_diff(power, s.power))      > .01) return false; // 1%
            if (dQt > dQtmp) dQtmp = dQt;

            if ((dQt=rel_diff(delay, s.delay))      > .01) return false; // 1%
            if (tight == 1 && dQt                  > .001) return false; // 0.1 %
            if (tight == 2 && dQt                 > .0001) return false; // 0.01 %
            if (tight == 3 && dQt                   > 0.0) return false; // exact
            if (dQt > dQtmp) dQtmp = dQt;
            return true;
    }
    return false;
}

// parse a SETI@home result file
//
int SAH_RESULT::parse_file(FILE* f) {
    char buf[1024];
    SIGNAL s;
    double d;
    int i, line = 0;

    num_signals = 0;

    memset(&s, 0, sizeof(s));
    while (fgets(buf, 256, f)) {
        line++;

        if (match_tag(buf, "<spike>")) {
            memset(&s, 0, sizeof(s));
            s.type = SIGNAL_TYPE_SPIKE;
            s.len_prof = line; // Using len_prof to save line number from result file
            num_signals++;

        } else if (match_tag(buf, "<best_spike>")) {
            memset(&s, 0, sizeof(s));
            s.type = SIGNAL_TYPE_BEST_SPIKE;
            s.len_prof = line;

        } else if (match_tag(buf, "<autocorr>")) {
            memset(&s, 0, sizeof(s));
            s.type = SIGNAL_TYPE_AUTOCORR;
            s.len_prof = line;
            num_signals++;

        } else if (match_tag(buf, "<best_autocorr>")) {
            memset(&s, 0, sizeof(s));
            s.type = SIGNAL_TYPE_BEST_AUTOCORR;
            s.len_prof = line;

        } else if (match_tag(buf, "<gaussian>")) {
            memset(&s, 0, sizeof(s));
            s.type = SIGNAL_TYPE_GAUSSIAN;
            s.len_prof = line;
            num_signals++;

        } else if (match_tag(buf, "<best_gaussian>")) {
            memset(&s, 0, sizeof(s));
            s.type = SIGNAL_TYPE_BEST_GAUSSIAN;
            s.len_prof = line;

        } else if (match_tag(buf, "<pulse>")) {
            memset(&s, 0, sizeof(s));
            s.type = SIGNAL_TYPE_PULSE;
            s.len_prof = line;
            num_signals++;

        } else if (match_tag(buf, "<best_pulse>")) {
            memset(&s, 0, sizeof(s));
            s.type = SIGNAL_TYPE_BEST_PULSE;
            s.len_prof = line;

        } else if (match_tag(buf, "<triplet>")) {
            memset(&s, 0, sizeof(s));
            s.type = SIGNAL_TYPE_TRIPLET;
            s.len_prof = line;
            num_signals++;

        } else if (match_tag(buf, "<best_triplet>")) {
            memset(&s, 0, sizeof(s));
            s.type = SIGNAL_TYPE_BEST_TRIPLET;
            s.len_prof = line;

        } else if (parse_double(buf, "<power>", d)) {
            s.power = d;
        } else if (parse_double(buf, "<period>", d)) {
            s.period = d;
        } else if (parse_double(buf, "<peak>", d)) {
            s.peak = d;
        } else if (parse_double(buf, "<mean>", d)) {
            s.mean = d;
        }  else if (parse_double(buf, "<ra>", d)) {
            s.ra = d;
        } else if (parse_double(buf, "<decl>", d)) {
            s.decl = d;
        } else if (parse_double(buf, "<time>", d)) {
            s.time = d;
        } else if (parse_double(buf, "<freq>", d)) {
            s.freq = d;
        }  else if (parse_double(buf, "<sigma>", d)) {
            s.sigma = d;
        } else if (parse_double(buf, "<chisqr>", d)) {
            s.chisqr = d;
        } else if (parse_double(buf, "<max_power>", d)) {
            s.max_power = d;
        } else if (parse_double(buf, "<peak_power>", d)) {
            s.peak_power = d;
        } else if (parse_double(buf, "<mean_power>", d)) {
            s.mean_power = d;
        } else if (parse_double(buf, "<score>", d)) {
            s.score = d;
        } else if (parse_double(buf, "<snr>", d)) {
            s.snr = d;
        } else if (parse_double(buf, "<thresh>", d)) {
            s.thresh = d;
        } else if (parse_double(buf, "<chirp_rate>", d)) {
            s.chirp_rate = d;
        } else if (parse_double(buf, "<delay>", d)) {
            s.delay = d;

        } else if (parse_int(buf, "<fft_len>", i)) {
            s.fft_len = i;

        } else if (match_tag(buf, "</spike>")) {
            if (s.type != SIGNAL_TYPE_SPIKE) {
                return -1;
            }
            signals.push_back(s);

        } else if (match_tag(buf, "</best_spike>")) {
            if (s.type != SIGNAL_TYPE_BEST_SPIKE) {
                return -1;
            }
            signals.push_back(s);

        } else if (match_tag(buf, "</autocorr>")) {
            if (s.type != SIGNAL_TYPE_AUTOCORR) {
                return -1;
            }
            signals.push_back(s);

        } else if (match_tag(buf, "</best_autocorr>")) {
            if (s.type != SIGNAL_TYPE_BEST_AUTOCORR) {
                return -1;
            }
            signals.push_back(s);

        } else if (match_tag(buf, "</gaussian>")) {
            if (s.type != SIGNAL_TYPE_GAUSSIAN) {
                return -1;
            }
            signals.push_back(s);

        } else if (match_tag(buf, "</best_gaussian>")) {
            if (s.type != SIGNAL_TYPE_BEST_GAUSSIAN) {
                return -1;
            }
            signals.push_back(s);

        } else if (match_tag(buf, "</pulse>")) {
            if (s.type != SIGNAL_TYPE_PULSE) {
                return -1;
            }
            signals.push_back(s);

        } else if (match_tag(buf, "</best_pulse>")) {
            if (s.type != SIGNAL_TYPE_BEST_PULSE) {
                return -1;
            }
            signals.push_back(s);

        } else if (match_tag(buf, "</triplet>")) {
            if (s.type != SIGNAL_TYPE_TRIPLET) {
                return -1;
            }
            signals.push_back(s);

        } else if (match_tag(buf, "</best_triplet>")) {
            if (s.type != SIGNAL_TYPE_BEST_TRIPLET) {
                return -1;
            }
            signals.push_back(s);

        }
    }

    return 0;
}

// return true if the given signal is roughly equal to a signal
// from the result
//
bool SAH_RESULT::has_roughly_equal_signal(SIGNAL& s) {
    unsigned int i;
    for (i=0; i<signals.size(); i++) {
        if (signals[i].roughly_equal(s, 0)) return true;
    }
    return false;
}

// return true if the given signal is nearly equal to a signal
// from the result (somewhat tighter check than above)
//
bool SAH_RESULT::has_nearly_equal_signal(SIGNAL& s, int tight) {
    unsigned int i;
    for (i=0; i<signals.size(); i++) {
        if (signals[i].roughly_equal(s, tight)) return true;
    }
    return false;
}

// return true if each signal from each result is roughly equal
// to a signal from the other result, and find the worst quality
// of the best matches.
//
bool SAH_RESULT::strongly_similar(SAH_RESULT& s) {
    unsigned int i, j;
    for (i=0; i<s.signals.size(); i++) {
        s.signals[i].checked = false;
    }
    for (i=0; i<signals.size(); i++) {
        if (!s.has_roughly_equal_signal(signals[i])) return false;
    }
    for (i=0; i<s.signals.size(); i++) {
        if (s.signals[i].checked) continue;
        if (!has_roughly_equal_signal(s.signals[i])) return false;
    }
    // Results are strongly similar, now check best matches, all parameters.
    // (Don't show low quality from a poor match if there's a better one)
    for (i=0; i<signals.size(); i++) {
        double dQmin = __DBL_MAX__;
        for (j=0; j<s.signals.size(); j++) {
            if (signals[i].roughly_equal(s.signals[j], 0)) {
                if (dQtmp < dQmin) dQmin = dQtmp;
            }
        }
        if (dQmin > dQfin) dQfin = dQmin;
    }
    return true;
}

// Return true if at least half the signals (and at least one)
// from each result are roughly equal to a signal from the other result.
//
// This is also called in some cases when it is already known that all
// the signals have matches, to generate the table indicating how good
// the matches are.
//
bool SAH_RESULT::weakly_similar(SAH_RESULT& s) {
    unsigned int n1, n2;
    unsigned int m1, m2;
    unsigned int i, j;
    unsigned int rA [11][10]; // 10 types + totals, 10 counts
    vector<int> R1lines;
    vector<int> R2lines;

    for (i = 0; i < 11; i++) {
        for (j = 0; j < 10; j++) {
            rA[i][j] = 0;
        }
    }

    n1 = signals.size();
    n2 = s.signals.size();

    for (i=0; i<signals.size(); i++) {
        j = signals[i].type;
        if (s.has_nearly_equal_signal(signals[i], 3)) rA[j][0]++;
        if (s.has_nearly_equal_signal(signals[i], 2)) rA[j][1]++;
        if (s.has_nearly_equal_signal(signals[i], 1)) rA[j][2]++;
    }
    m1 = 0;
    for (i=0; i<signals.size(); i++) {
        j = signals[i].type;
        if (s.has_roughly_equal_signal(signals[i])) {
            m1++;
            rA[j][3]++;
        } else {
            rA[j][4]++;
            if (R1lines.size() < 101) R1lines.push_back(signals[i].len_prof);
        }
    }

    for (i=0; i<s.signals.size(); i++) {
        j = s.signals[i].type;
        if (has_nearly_equal_signal(s.signals[i], 3)) rA[j][5]++;
        if (has_nearly_equal_signal(s.signals[i], 2)) rA[j][6]++;
        if (has_nearly_equal_signal(s.signals[i], 1)) rA[j][7]++;
    }
    m2 = 0;
    for (i=0; i<s.signals.size(); i++) {
        j = s.signals[i].type;
        if (has_roughly_equal_signal(s.signals[i])) {
            m2++;
            rA[j][8]++;
        } else {
            rA[j][9]++;
            if (R2lines.size() < 101) R2lines.push_back(s.signals[i].len_prof);
        }
    }

    for (i = 0; i < 10; i++) {
        for (j = 0; j < 10; j++) {
            rA[10][j] += rA[i][j];
        }
    }

    cout << "                ------------- R1:R2 ------------     ------------- R2:R1 ------------" << endl;
    cout << "                Exact  Super  Tight  Good    Bad     Exact  Super  Tight  Good    Bad" << endl;
    cout << "        Spike"<<setw(7)<<rA[0][0]<<setw(7)<<rA[0][1]<<setw(7)<<rA[0][2]<<setw(7)<<rA[0][3]<<setw(7)<<rA[0][4]<<setw(9)<<rA[0][5]<<setw(7)<<rA[0][6]<<setw(7)<<rA[0][7]<<setw(7)<<rA[0][8]<<setw(7)<<rA[0][9] << endl;
    if (show_autocorrs)
    cout << "     Autocorr"<<setw(7)<<rA[1][0]<<setw(7)<<rA[1][1]<<setw(7)<<rA[1][2]<<setw(7)<<rA[1][3]<<setw(7)<<rA[1][4]<<setw(9)<<rA[1][5]<<setw(7)<<rA[1][6]<<setw(7)<<rA[1][7]<<setw(7)<<rA[1][8]<<setw(7)<<rA[1][9] << endl;
    cout << "     Gaussian"<<setw(7)<<rA[2][0]<<setw(7)<<rA[2][1]<<setw(7)<<rA[2][2]<<setw(7)<<rA[2][3]<<setw(7)<<rA[2][4]<<setw(9)<<rA[2][5]<<setw(7)<<rA[2][6]<<setw(7)<<rA[2][7]<<setw(7)<<rA[2][8]<<setw(7)<<rA[2][9] << endl;
    cout << "        Pulse"<<setw(7)<<rA[3][0]<<setw(7)<<rA[3][1]<<setw(7)<<rA[3][2]<<setw(7)<<rA[3][3]<<setw(7)<<rA[3][4]<<setw(9)<<rA[3][5]<<setw(7)<<rA[3][6]<<setw(7)<<rA[3][7]<<setw(7)<<rA[3][8]<<setw(7)<<rA[3][9] << endl;
    cout << "      Triplet"<<setw(7)<<rA[4][0]<<setw(7)<<rA[4][1]<<setw(7)<<rA[4][2]<<setw(7)<<rA[4][3]<<setw(7)<<rA[4][4]<<setw(9)<<rA[4][5]<<setw(7)<<rA[4][6]<<setw(7)<<rA[4][7]<<setw(7)<<rA[4][8]<<setw(7)<<rA[4][9] << endl;
    cout << "   Best Spike"<<setw(7)<<rA[5][0]<<setw(7)<<rA[5][1]<<setw(7)<<rA[5][2]<<setw(7)<<rA[5][3]<<setw(7)<<rA[5][4]<<setw(9)<<rA[5][5]<<setw(7)<<rA[5][6]<<setw(7)<<rA[5][7]<<setw(7)<<rA[5][8]<<setw(7)<<rA[5][9] << endl;
    if (show_autocorrs)
    cout << "Best Autocorr"<<setw(7)<<rA[6][0]<<setw(7)<<rA[6][1]<<setw(7)<<rA[6][2]<<setw(7)<<rA[6][3]<<setw(7)<<rA[6][4]<<setw(9)<<rA[6][5]<<setw(7)<<rA[6][6]<<setw(7)<<rA[6][7]<<setw(7)<<rA[6][8]<<setw(7)<<rA[6][9] << endl;
    cout << "Best Gaussian"<<setw(7)<<rA[7][0]<<setw(7)<<rA[7][1]<<setw(7)<<rA[7][2]<<setw(7)<<rA[7][3]<<setw(7)<<rA[7][4]<<setw(9)<<rA[7][5]<<setw(7)<<rA[7][6]<<setw(7)<<rA[7][7]<<setw(7)<<rA[7][8]<<setw(7)<<rA[7][9] << endl;
    cout << "   Best Pulse"<<setw(7)<<rA[8][0]<<setw(7)<<rA[8][1]<<setw(7)<<rA[8][2]<<setw(7)<<rA[8][3]<<setw(7)<<rA[8][4]<<setw(9)<<rA[8][5]<<setw(7)<<rA[8][6]<<setw(7)<<rA[8][7]<<setw(7)<<rA[8][8]<<setw(7)<<rA[8][9] << endl;
    cout << " Best Triplet"<<setw(7)<<rA[9][0]<<setw(7)<<rA[9][1]<<setw(7)<<rA[9][2]<<setw(7)<<rA[9][3]<<setw(7)<<rA[9][4]<<setw(9)<<rA[9][5]<<setw(7)<<rA[9][6]<<setw(7)<<rA[9][7]<<setw(7)<<rA[9][8]<<setw(7)<<rA[9][9] << endl;
    cout << "                ----   ----   ----   ----   ----     ----   ----   ----   ----   ----" << endl;
    cout << "             "<<setw(7)<<rA[10][0]<<setw(7)<<rA[10][1]<<setw(7)<<rA[10][2]<<setw(7)<<rA[10][3]<<setw(7)<<rA[10][4]<<setw(9)<<rA[10][5]<<setw(7)<<rA[10][6]<<setw(7)<<rA[10][7]<<setw(7)<<rA[10][8]<<setw(7)<<rA[10][9] << endl;
    cout << "\n";

    if (R1lines.size()) {
        cout << "Unmatched signal(s) in R1 at line(s)";
        for (i=0; i<R1lines.size(); i++) cout << " " << R1lines[i];
        cout << "\n";
    }
    if (R2lines.size()) {
        cout << "Unmatched signal(s) in R2 at line(s)";
        for (i=0; i<R2lines.size(); i++) cout << " " << R2lines[i];
        cout << "\n";
    }

    // If dQfin is at its intialization value of -1.0, results are not strongly similar.
    // In that case check any matched signals now, all parameters, and output the Q
    // value here (unless none match, of course).
    if (dQfin == -1.0) {
        for (i=0; i<signals.size(); i++) {
            double dQmin = __DBL_MAX__;
            for (j=0; j<s.signals.size(); j++) {
                if (signals[i].roughly_equal(s.signals[j], 0)) {
                    if (dQtmp < dQmin) dQmin = dQtmp;
                }
            }
            if (dQmin != __DBL_MAX__ && dQmin > dQfin) dQfin = dQmin;
        }
        if (dQfin > .01)
            cout << "For R1:R2 matched signals only, Q= " << "????" << endl;
        else if (dQfin >= 0.0) {
            cout.precision(4);
            cout << "For R1:R2 matched signals only, Q= " << showpoint << 100.0-dQfin*10000.0 << "%" << endl;
        }
    }

    if (m1 == 0) return false;
    if (m2 == 0) return false;
    if (m1 < (n1+1)/2) return false;
    if (m2 < (n2+1)/2) return false;
    return true;
}

bool SAH_RESULT::bad_values() {

	return false;
}

void SAH_RESULT::dump_signals(){
  int i;

  for (i = 0; i < signals.size(); i++){
    switch(signals[i].type){
    case SIGNAL_TYPE_SPIKE:
      cout << "spike" << endl;
      break;
    case SIGNAL_TYPE_BEST_SPIKE:
      cout << "best spike" << endl;
      break;
      // I omit the rest :)
    }
  }
}
