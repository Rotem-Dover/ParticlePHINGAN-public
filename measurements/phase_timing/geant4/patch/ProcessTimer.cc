//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
/// \file ProcessTimer.cc
/// \brief Implementation of ProcessTimer (see header).
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#include "ProcessTimer.hh"

#include <iomanip>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ProcessTimer& ProcessTimer::Instance()
{
  static ProcessTimer instance;
  return instance;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ProcessTimer::BeginRun()
{
  fAlongGPIL_t = fPostGPIL_t = fAlongDoIt_t = fPostDoIt_t = 0;
  fAlongGPIL_n = fPostGPIL_n = fAlongDoIt_n = fPostDoIt_n = 0;
  fRunWall_s = 0.0;
  fRunStart = clock::now();
  fRunStartTicks = Now();
}

void ProcessTimer::EndRun()
{
  const ticks dt = Now() - fRunStartTicks;
  fRunWall_s = std::chrono::duration<double>(clock::now() - fRunStart).count();
  fTicksPerSecond = fRunWall_s > 0.0 ? static_cast<double>(dt) / fRunWall_s : 0.0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

double ProcessTimer::MeasureTimerCost_ticks()
{
  // Cost of one (Now() - Now()) pair, i.e. what every timed call pays on
  // top of the wrapped work. Median of a few batches to dodge a stray stall.
  const int kBatches = 5;
  const long kIters = 200000;
  double est[kBatches];
  for (int b = 0; b < kBatches; ++b) {
    volatile ticks sink = 0;
    const ticks t0 = Now();
    for (long i = 0; i < kIters; ++i) {
      const ticks a = Now();
      const ticks c = Now();
      sink = sink + (c - a);
    }
    const ticks t1 = Now();
    est[b] = static_cast<double>(t1 - t0) / kIters;
    (void)sink;
  }
  for (int i = 1; i < kBatches; ++i)
    for (int j = i; j > 0 && est[j - 1] > est[j]; --j) std::swap(est[j - 1], est[j]);
  return est[kBatches / 2];
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ProcessTimer::Report(std::ostream& os, G4int nEvents) const
{
  std::ios::fmtflags f0(os.flags());
  if (!fEnabled) {
    os << "\n ProcessTimer: off. beamOn wall = " << std::fixed << std::setprecision(4)
       << fRunWall_s << " s (" << std::setprecision(3)
       << 1e3 * fRunWall_s / std::max(1, nEvents) << " ms/event)\n";
    os.flags(f0);
    return;
  }

  const double tps = fTicksPerSecond > 0.0 ? fTicksPerSecond : 1.0;
  auto sec = [tps](ticks t) { return static_cast<double>(t) / tps; };
  const double timerCost = MeasureTimerCost_ticks() / tps;
  const unsigned long long nCalls = fAlongGPIL_n + fPostGPIL_n + fAlongDoIt_n + fPostDoIt_n;
  const double timerTotal = timerCost * static_cast<double>(nCalls);

  const double stepLen_raw = sec(fAlongGPIL_t + fPostGPIL_t);
  const double cont_raw    = sec(fAlongDoIt_t);
  const double sec_raw     = sec(fPostDoIt_t);
  const double phases_raw  = stepLen_raw + cont_raw + sec_raw;
  const double other_raw   = fRunWall_s - phases_raw;

  // Corrected: subtract the timer's own cost per call from each bucket and
  // from the wall (the wrapped process would not have paid it).
  const double stepLen_c = stepLen_raw - timerCost * (fAlongGPIL_n + fPostGPIL_n);
  const double cont_c    = cont_raw    - timerCost * fAlongDoIt_n;
  const double sec_c     = sec_raw     - timerCost * fPostDoIt_n;
  const double wall_c    = fRunWall_s  - timerTotal;
  const double phases_c  = stepLen_c + cont_c + sec_c;
  const double other_c   = wall_c - phases_c;

  auto pct = [](double x, double tot) { return tot > 0.0 ? 100.0 * x / tot : 0.0; };

  std::ios::fmtflags f(os.flags());
  os << "\n ================= ProcessTimer (hIoni phases) =================\n"
     << " events          : " << nEvents << "\n"
     << " beamOn wall     : " << std::fixed << std::setprecision(4) << fRunWall_s << " s"
     << "   (" << std::setprecision(3) << 1e3 * fRunWall_s / std::max(1, nEvents) << " ms/event)\n"
     << " tick rate       : " << std::setprecision(1) << 1e-6 * fTicksPerSecond << " MHz (calibrated over the run)\n"
     << " timer cost      : " << std::setprecision(1) << 1e9 * timerCost << " ns/call x "
     << nCalls << " calls = " << std::setprecision(4) << timerTotal << " s ("
     << std::setprecision(2) << pct(timerTotal, fRunWall_s) << " % of wall)\n"
     << " calls           : alongGPIL " << fAlongGPIL_n << ", postGPIL " << fPostGPIL_n
     << ", alongDoIt " << fAlongDoIt_n << ", postDoIt " << fPostDoIt_n << "\n\n"
     << std::setprecision(4)
     << "  bucket                      raw [s]   share     corrected [s]   share\n"
     << "  step length (GPIL a+p)  " << std::setw(10) << stepLen_raw << std::setw(8) << std::setprecision(1) << pct(stepLen_raw, fRunWall_s) << " %"
     << std::setw(16) << std::setprecision(4) << stepLen_c << std::setw(8) << std::setprecision(1) << pct(stepLen_c, wall_c) << " %\n"
     << "  continuous  (AlongDoIt) " << std::setw(10) << std::setprecision(4) << cont_raw << std::setw(8) << std::setprecision(1) << pct(cont_raw, fRunWall_s) << " %"
     << std::setw(16) << std::setprecision(4) << cont_c << std::setw(8) << std::setprecision(1) << pct(cont_c, wall_c) << " %\n"
     << "  secondary   (PostDoIt)  " << std::setw(10) << std::setprecision(4) << sec_raw << std::setw(8) << std::setprecision(1) << pct(sec_raw, fRunWall_s) << " %"
     << std::setw(16) << std::setprecision(4) << sec_c << std::setw(8) << std::setprecision(1) << pct(sec_c, wall_c) << " %\n"
     << "  other (kernel/transport)" << std::setw(10) << std::setprecision(4) << other_raw << std::setw(8) << std::setprecision(1) << pct(other_raw, fRunWall_s) << " %"
     << std::setw(16) << std::setprecision(4) << other_c << std::setw(8) << std::setprecision(1) << pct(other_c, wall_c) << " %\n"
     << "  hIoni phases total      " << std::setw(10) << std::setprecision(4) << phases_raw << std::setw(8) << std::setprecision(1) << pct(phases_raw, fRunWall_s) << " %"
     << std::setw(16) << std::setprecision(4) << phases_c << std::setw(8) << std::setprecision(1) << pct(phases_c, wall_c) << " %\n"
     << "  of hIoni only: step length " << std::setprecision(1) << pct(stepLen_c, phases_c)
     << " %, continuous " << pct(cont_c, phases_c) << " %, secondary " << pct(sec_c, phases_c) << " %\n"
     << "  (the per-call correction is the isolated counter-pair cost, a\n"
     << "   same-size ESTIMATE of the in-situ wrapper overhead -- measured\n"
     << "   12.4 ns in situ vs 15.7 isolated on a Xeon Gold 6354; prefer\n"
     << "   (wall_on - wall_off)/calls from a paired run on the same core)\n"
     << " ================================================================\n";
  os.flags(f);
}
