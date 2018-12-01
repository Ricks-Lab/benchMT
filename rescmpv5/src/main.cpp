/*
 * main.cpp for rescmpv
 *
 * by Tetsuji Maverick Rai
 * modified by Josef W. Segur
 *
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "sah_result.h"


using namespace std;

FILE* fp0;
FILE* fp1;

SAH_RESULT SahRes;
SAH_RESULT SahRes1;

double dQfin = -1.0;
double dQtmp;
double dTThresh = 0.0001;
bool show_autocorrs = true;

int main(int argc, char** argv){
  bool hasAutocorrs1=false, hasAutocorrs2=false;
  if (argc < 3 || argc > 4){
    cerr
      << "Usage: " << argv[0] <<" {result.sah 1} {result.sah 2} [Qnn.n]" << endl
      << endl
      <<"  \"strongly similar\" means all the signals are similar." << endl
      <<"  \"weakly similar\" means at least half the signals (and at least one)" << endl
      <<"  from each result are roughly equal to a signal from the other result." << endl
      << endl
      <<"  The optional Qnn.n third argument specifies the maximum Q value at which" << endl
      <<"  the table of signal types is shown for strongly similar results. The" << endl
      <<"  numerical part of the argument is evaluated as a double float and is" << endl
      <<"  ignored if outside the 0 through 100 range." << endl
      <<"    Q0 would suppress the table" << endl
      <<"    Q100 would show the table for all results" << endl
      <<"    Q99 is the default" << endl;

    exit (1);
  }

  if ((fp0 = fopen(argv[1],"r")) == NULL){
    cerr << "Cannot open " << argv[1] << endl;
    exit (1);
  }

  if ((fp1 = fopen(argv[2],"r")) == NULL){
    cerr << "Cannot open " << argv[2] << endl;
    exit (1);
  }

  if(SahRes.parse_file(fp0)){
    cerr << "Error parsing " << argv[1] << endl;
    exit (1);
  } else {
    unsigned int i;
    for (i=0; i<SahRes.signals.size(); i++) {
      if (SahRes.signals[i].type == SIGNAL_TYPE_AUTOCORR
        || SahRes.signals[i].type == SIGNAL_TYPE_BEST_AUTOCORR) {
          hasAutocorrs1 = true;
          break;
      }
    }
  }

  if(SahRes1.parse_file(fp1)){
    cerr << "Error parsing " << argv[2] << endl;
    exit (1);
  } else {
    unsigned int i;
    for (i=0; i<SahRes1.signals.size(); i++) {
      if (SahRes1.signals[i].type == SIGNAL_TYPE_AUTOCORR
        || SahRes1.signals[i].type == SIGNAL_TYPE_BEST_AUTOCORR) {
          hasAutocorrs2 = true;
          break;
       }
    }
  }

  if (!hasAutocorrs1 && !hasAutocorrs2) show_autocorrs = false;

  if (argc == 4 && !strncmp(argv[3],"Q", 1)){
    char buf[256];
    strncpy (buf, argv[3], 255);
    *buf = ' ';
    double dTmp = strtod(buf, NULL);
    if (dTmp >= 0.0 && dTmp <= 100.0) dTThresh = (100.0 - dTmp) / 10000.0; 
  }

#ifdef DUMP
  SahRes.dump_signals();
  SahRes1.dump_signals();
#endif // DUMP

  if (SahRes.strongly_similar(SahRes1) == true){
    if (dQfin >= dTThresh) {
        bool b = SahRes.weakly_similar(SahRes1); // Show table if Q <= threshold
    }
    if (dQfin > .01) 
        cout << "Result      : Strongly similar,  Q= " << "????" << endl;
    else {
        cout.precision(4);
        cout << "Result      : Strongly similar,  Q= " << showpoint << 100.0-dQfin*10000.0 << "%" << endl;
    }
    return 0;
  }

  if (SahRes.weakly_similar(SahRes1) == true){
    cout << "Result      : Weakly similar." << endl;
    return 0;
  }

  cout << "Result      : Different." << endl;

  return 0;
}
