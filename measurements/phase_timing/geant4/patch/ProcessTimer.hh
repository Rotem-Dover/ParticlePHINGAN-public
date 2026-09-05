//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
/// \file ProcessTimer.hh
/// \brief Per-phase wall-time accumulator for one timed G4VProcess.
///
/// Singleton that the TimedProcess wrapper writes into and RunAction reads
/// out at end of run. Buckets mirror the G4VProcess phase structure:
///   AlongStepGPIL + PostStepGPIL  -> "step length"
///   AlongStepDoIt                 -> "continuous loss"
///   PostStepDoIt                  -> "secondary"
/// Everything else in the beamOn wall (stepping manager, Transportation,
/// StepMax, tracking/stacking, user actions) is reported as "other".
///
/// Serial run manager only: the accumulators are plain doubles.
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#ifndef ProcessTimer_h
#define ProcessTimer_h 1

#include "globals.hh"
#include <chrono>
#include <ostream>

class ProcessTimer
{
public:
  using clock = std::chrono::steady_clock;
  using ticks = unsigned long long;

  static ProcessTimer& Instance();

  /// Cheapest monotonic counter on the host: rdtsc (x86-64, constant-rate
  /// TSC on any Xeon of the last decade), cntvct_el0 (aarch64), else
  /// steady_clock nanoseconds. Converted to seconds with a rate calibrated
  /// against steady_clock over the run itself, so no frequency is assumed.
  static inline ticks Now();

  void   SetEnabled(G4bool on) { fEnabled = on; }
  G4bool Enabled() const       { return fEnabled; }

  // Called by RunAction around the event loop (beamOn).
  void BeginRun();
  void EndRun();

  // Called by TimedProcess; `dt` in ticks.
  void AddAlongGPIL(ticks dt) { fAlongGPIL_t += dt; ++fAlongGPIL_n; }
  void AddPostGPIL (ticks dt) { fPostGPIL_t  += dt; ++fPostGPIL_n;  }
  void AddAlongDoIt(ticks dt) { fAlongDoIt_t += dt; ++fAlongDoIt_n; }
  void AddPostDoIt (ticks dt) { fPostDoIt_t  += dt; ++fPostDoIt_n;  }

  // Prints the table (raw and timer-overhead-corrected) to `os`.
  void Report(std::ostream& os, G4int nEvents) const;

private:
  ProcessTimer() = default;

  // One (Now() - Now()) pair, in ticks, measured in place at report time.
  static double MeasureTimerCost_ticks();

  G4bool fEnabled = false;
  clock::time_point fRunStart{};
  ticks  fRunStartTicks = 0;
  double fRunWall_s = 0.0;
  double fTicksPerSecond = 0.0;   // calibrated in EndRun

  ticks fAlongGPIL_t = 0, fPostGPIL_t = 0, fAlongDoIt_t = 0, fPostDoIt_t = 0;
  unsigned long long fAlongGPIL_n = 0, fPostGPIL_n = 0, fAlongDoIt_n = 0, fPostDoIt_n = 0;
};

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
inline ProcessTimer::ticks ProcessTimer::Now() { return __rdtsc(); }
#elif defined(__aarch64__)
inline ProcessTimer::ticks ProcessTimer::Now()
{
  ticks v; asm volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
}
#else
inline ProcessTimer::ticks ProcessTimer::Now()
{
  return static_cast<ticks>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      clock::now().time_since_epoch()).count());
}
#endif

#endif
